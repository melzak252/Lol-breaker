"""Boundary contracts for EV policy selection and delayed-settlement accounting."""

import math

import pandas as pd
import pytest

from src.analysis import ev_approach_policies as policies


def opportunity(**overrides):
    row = {
        "bet_id": "bet-1",
        "row_id": "row-1",
        "golgg_match_id": "match-1",
        "cohort": "historical",
        "horizon": "OPEN",
        "date": "2026-01-01",
        "decision_at": "2026-01-01T10:00:00Z",
        "start_at": "2026-01-01T12:00:00Z",
        "label_available_at": "2026-01-02T12:00:00Z",
        "y": 1,
        "p": 0.60,
        "q": 0.55,
        "odds_a": 2.20,
        "odds_b": 3.00,
        "bookmaker_a": "book-a",
        "bookmaker_b": "book-b",
        "quote_id_a": "quote-a",
        "quote_id_b": "quote-b",
        "quote_at_a": "2026-01-01T09:55:00Z",
        "quote_at_b": "2026-01-01T09:55:00Z",
        "best_of": 3,
        "league": "Regional League",
        "information_status": "available",
        "strict_prior_day": False,
        "is_common": True,
        "close_q": 0.99,
        "close_odds_a": 20.0,
        "close_odds_b": 1.01,
        "close_available_at": "2026-01-02T13:00:00Z",
        "sigma_z": 0.20,
        "p_std": 0.04,
        "p_q10": 0.52,
        "p_q90": 0.70,
        "model_year": 2025,
        "variant": "seed_only",
    }
    row.update(overrides)
    return row


def test_policies_use_independent_side_specific_probability_arithmetic():
    row = opportunity()
    p_a, p_b = row["p"], 1.0 - row["p"]
    logistic = lambda x: 1.0 / (1.0 + math.exp(-x))
    logit = lambda p: math.log(p / (1.0 - p))

    assert policies.policy_probability(p_a, "P0", side="A") == pytest.approx(0.60)
    assert policies.policy_probability(p_b, "P0", side="B") == pytest.approx(0.40)
    assert policies.policy_probability(p_a, "P1", side="A") == pytest.approx(
        logistic(logit(0.60) - 0.75 * 0.15)
    )
    assert policies.policy_probability(p_a, "P2", side="A", sigma_z=0.20) == pytest.approx(
        logistic(logit(0.60) - 0.75 * 0.20)
    )
    assert policies.policy_probability(p_a, "P3", side="A", p_std=0.04) == pytest.approx(
        0.60 - 1.5 * 0.04
    )
    assert policies.policy_probability(
        p_a, "P4", side="A", p_q10=0.52, p_q90=0.70
    ) == pytest.approx(0.52)
    assert policies.policy_probability(
        p_b, "P4", side="B", p_q10=0.52, p_q90=0.70
    ) == pytest.approx(1.0 - 0.70)
    assert policies.policy_probability(
        p_a, "P5", side="A", sigma_z=0.20, p_std=0.99, threshold=0.20
    ) == pytest.approx(0.60)
    assert policies.policy_probability(
        p_a, "P5", side="A", sigma_z=0.200001, p_std=0.0, threshold=0.20
    ) is None


def test_lower_probability_policies_stay_finite_and_never_exceed_mean():
    for p in (1e-9, 0.01, 0.5, 0.99, 1.0 - 1e-9):
        for method, kwargs in (
            ("P1", {}),
            ("P2", {"sigma_z": 100.0}),
            ("P3", {"p_std": 100.0}),
            ("P4", {"p_q10": 0.0, "p_q90": 1.0}),
        ):
            result = policies.policy_probability(p, method, side="A", **kwargs)
            assert result is not None and math.isfinite(result)
            assert 0.0 <= result <= p


def test_raw_selection_chooses_one_highest_ev_side_using_offered_prices():
    frame = pd.DataFrame([
        opportunity(p=0.60, odds_a=2.0, odds_b=3.2, q=0.5),
        opportunity(row_id="row-2", bet_id="bet-2", golgg_match_id="match-2",
                    p=0.40, odds_a=3.2, odds_b=2.0, q=0.5),
    ])
    selected = policies.select_policy_bets(
        frame, probability_col="p", policy="P0", tax_rate=0.0,
        min_ev=0.05, qualification="raw",
    )
    assert len(selected) == 2
    assert selected["side"].tolist() == ["B", "A"]
    assert selected["odds"].tolist() == [3.2, 3.2]


def test_raw_selection_ties_choose_side_a_and_keep_source_price_and_market_data():
    selected = policies.select_policy_bets(
        pd.DataFrame([opportunity(p=0.5, q=0.5, odds_a=2.4, odds_b=2.4)]),
        tax_rate=0.0, min_ev=0.05, qualification="raw",
    )

    assert selected["side"].tolist() == ["A"]
    assert selected.loc[0, "p"] == pytest.approx(0.5)
    assert selected.loc[0, "p_for_ev"] == pytest.approx(0.5)
    assert selected.loc[0, "odds"] == pytest.approx(2.4)
    assert selected.loc[0, "bookmaker"] == "book-a"
    assert selected.loc[0, "quote_id"] == "quote-a"
    assert selected.loc[0, "q"] == pytest.approx(0.5)
    assert selected.loc[0, "p_mean_a"] == pytest.approx(0.5)


def test_application_qualification_uses_real_market_probability_and_league():
    # At odds 3.0, this divergence passes the ordinary league gate but is rejected
    # by the Tier-1 market-gap gate. A fabricated or dropped q/league changes this.
    regional = pd.DataFrame([opportunity(p=0.46, q=0.32, odds_a=2.6, odds_b=1.4)])
    tier_one = pd.DataFrame([opportunity(p=0.46, q=0.32, league="LCK",
                                         odds_a=2.6, odds_b=1.4)])
    options = dict(probability_col="p", policy="P0", tax_rate=0.0,
                   min_ev=0.05, qualification="application")
    regional_selected = policies.select_policy_bets(regional, **options)
    assert len(regional_selected) == 1
    assert regional_selected.loc[0, "odds"] == pytest.approx(2.6)
    assert regional_selected.loc[0, "q"] == pytest.approx(0.32)
    assert regional_selected.loc[0, "league"] == "Regional League"
    assert regional_selected.loc[0, "date"] == "2026-01-01"
    assert regional_selected.loc[0, "best_of"] == 3
    assert regional_selected.loc[0, "competition_tier"] is not None
    assert policies.select_policy_bets(tier_one, **options).empty


@pytest.mark.parametrize("qualification", ["raw", "application"])
@pytest.mark.parametrize("policy", ["P0", "P5"])
def test_mean_policy_keeps_mean_ev_and_no_conservative_evidence_in_public_records(
    qualification, policy,
):
    selected = policies.select_policy_bets(
        pd.DataFrame([opportunity(odds_b=1.4)]),
        policy=policy, qualification=qualification, tax_rate=0.0, min_ev=0.05,
        cal_thresholds={("seed_only", 2025): 0.20},
    )

    assert selected["side"].tolist() == ["A"]
    assert selected.loc[0, "prob_conservative"] is None
    assert selected.loc[0, "prob_for_ev"] == selected.loc[0, "p_for_ev"] == 0.60
    assert selected.loc[0, "uncertainty_basis"] == "mean"
    assert selected.loc[0, "uncertainty_reason"]
    assert selected.loc[0, "ev"] == pytest.approx(0.60 * 2.20 - 1.0)
    ledger, _ = policies.simulate_portfolio(
        selected, strategy="flat", initial_bankroll=100.0, flat_stake=10.0,
        tax_rate=0.0,
    )
    for field in ("prob_conservative", "prob_for_ev", "uncertainty_basis", "uncertainty_reason"):
        assert ledger.loc[0, field] == selected.loc[0, field]


@pytest.mark.parametrize("qualification", ["raw", "application"])
@pytest.mark.parametrize("policy,expected", [
    ("P1", 1.0 / (1.0 + math.exp(-(math.log(0.60 / 0.40) - 0.75 * 0.15)))),
    ("P2", 1.0 / (1.0 + math.exp(-(math.log(0.60 / 0.40) - 0.75 * 0.20)))),
    ("P3", 0.60 - 1.5 * 0.04),
    ("P4", 0.52),
])
def test_research_haircut_families_preserve_scores_without_claiming_measured_bounds(
    qualification, policy, expected,
):
    selected = policies.select_policy_bets(
        pd.DataFrame([opportunity(odds_b=1.4)]),
        policy=policy, qualification=qualification, tax_rate=0.0, min_ev=0.05,
    )

    assert selected["side"].tolist() == ["A"]
    assert selected.loc[0, "p"] == 0.60
    assert selected.loc[0, "prob_for_ev"] == pytest.approx(expected)
    assert selected.loc[0, "p_for_ev"] == pytest.approx(expected)
    assert selected.loc[0, "prob_conservative"] == pytest.approx(expected)
    assert selected.loc[0, "ev"] == pytest.approx(expected * 2.20 - 1.0)
    assert selected.loc[0, "uncertainty_basis"] == "assumed_research_haircut"
    assert selected.loc[0, "uncertainty_reason"]


@pytest.mark.parametrize("policy,inputs", [
    ("P1", {"p": 1.0}),
    ("P2", {"sigma_z": 0.0}),
    ("P3", {"p_std": 0.0}),
    ("P4", {"p_q10": 0.60, "p_q90": 0.60}),
])
def test_research_policy_identity_does_not_depend_on_score_being_below_mean(policy, inputs):
    selected = policies.select_policy_bets(
        pd.DataFrame([opportunity(odds_b=1.4, **inputs)]),
        policy=policy, qualification="raw", tax_rate=0.0, min_ev=0.05,
    )

    assert selected["side"].tolist() == ["A"]
    assert selected.loc[0, "prob_for_ev"] == pytest.approx(selected.loc[0, "p"])
    assert selected.loc[0, "prob_conservative"] == pytest.approx(selected.loc[0, "p"])
    assert selected.loc[0, "uncertainty_basis"] == "assumed_research_haircut"


@pytest.mark.parametrize("qualification", ["raw", "application"])
@pytest.mark.parametrize("policy,inputs,thresholds", [
    ("P2", {"sigma_z": None}, None),
    ("P3", {"p_std": None}, None),
    ("P4", {"p_q10": None}, None),
    ("P5", {"sigma_z": None}, {("seed_only", 2025): 0.20}),
    ("P5", {}, None),
    ("P5", {"sigma_z": 0.200001}, {("seed_only", 2025): 0.20}),
])
def test_risk_bound_cannot_supply_missing_policy_inputs_or_bypass_p5_abstention(
    qualification, policy, inputs, thresholds,
):
    selected = policies.select_policy_bets(
        pd.DataFrame([opportunity(
            odds_b=1.4, p_low_a=0.56, p_low_b=0.30,
            uncertainty_basis="measured_bounds", **inputs,
        )]),
        policy=policy, qualification=qualification, tax_rate=0.0, min_ev=0.05,
        cal_thresholds=thresholds,
    )

    assert selected.empty


def test_clv_summary_counts_only_observed_same_book_close_prices():
    frame = pd.DataFrame([opportunity(
        close_odds_a=2.0,
        close_bookmaker_a="book-a",
        close_available_at="2026-01-01T11:00:00Z",
    )])
    selected = policies.select_policy_bets(
        frame, policy="P0", tax_rate=0.0, min_ev=0.05, qualification="raw",
    )

    ledger, summary = policies.simulate_portfolio(
        selected, strategy="flat", initial_bankroll=1000.0,
        flat_stake=100.0, tax_rate=0.0,
    )

    assert selected.loc[0, "side"] == "A"
    assert selected.loc[0, "close_odds"] == pytest.approx(2.0)
    assert bool(selected.loc[0, "close_same_book"])
    assert ledger.loc[0, "placed"] is True
    assert summary["observed_clv_available"] is True
    assert summary["observed_clv_count"] == 1
    assert summary["mean_observed_clv_pct"] == pytest.approx(10.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_simultaneous_opposite_settlements_have_no_order_dependent_drawdown(reverse):
    bets = pd.DataFrame([
        {
            "bet_id": "win", "decision_at": "2026-01-01T09:00:00Z",
            "settle_at": "2026-01-01T12:00:00Z", "odds": 2.0, "won": True,
        },
        {
            "bet_id": "loss", "decision_at": "2026-01-01T09:00:00Z",
            "settle_at": "2026-01-01T12:00:00Z", "odds": 2.0, "won": False,
        },
    ])
    if reverse:
        bets = bets.iloc[::-1].reset_index(drop=True)

    ledger, summary = policies.simulate_portfolio(
        bets, strategy="flat", initial_bankroll=100.0, flat_stake=10.0,
        tax_rate=0.0,
    )

    assert ledger["pnl"].sum() == pytest.approx(0.0)
    assert summary["final_capital"] == pytest.approx(100.0)
    assert summary["max_drawdown"] == pytest.approx(0.0)


@pytest.mark.parametrize("close_time", [
    "2026-01-01T09:00:00Z",
    "2026-01-01T12:00:00Z",
    "2026-01-01T13:00:00Z",
])
def test_clv_close_outside_decision_to_event_start_window_is_unavailable(close_time):
    selected = policies.select_policy_bets(
        pd.DataFrame([opportunity(
            close_odds_a=2.0,
            close_bookmaker_a="book-a",
            close_available_at=close_time,
        )]),
        policy="P0", tax_rate=0.0, min_ev=0.05, qualification="raw",
    )

    ledger, summary = policies.simulate_portfolio(
        selected, strategy="flat", initial_bankroll=1000.0,
        flat_stake=100.0, tax_rate=0.0,
    )

    assert ledger.loc[0, "placed"] is True
    assert summary["observed_clv_count"] == 0
    assert summary["mean_observed_clv_pct"] is None


def test_p5_selection_preserves_runner_precomputed_calibration_provenance():
    frame = pd.DataFrame([opportunity(sigma_z=0.25, model_year=2025.0)])
    cal = pd.DataFrame([{
        "variant": "seed_only",
        "model_year": 2025,
        "sigma_z_q75": 0.30,
        "cal_rows": 4,
        "cal_row_ids": ["cal-a", "cal-b", "cal-c", "cal-d"],
    }])

    selected = policies.select_policy_bets(
        frame, policy="P5", tax_rate=0.0, min_ev=0.05,
        qualification="raw", cal_thresholds=cal,
    )

    assert selected["bet_id"].tolist() == ["bet-1"]
    assert selected.loc[0, "p5_threshold_source_ids"] == [
        "cal-a", "cal-b", "cal-c", "cal-d",
    ]
    assert selected.loc[0, "p5_threshold_n"] == 4


def test_close_information_cannot_change_earlier_open_bet_selection():
    frame = pd.DataFrame([opportunity()])
    changed_close = frame.assign(close_q=0.01, close_odds_a=1.01,
                                 close_odds_b=50.0,
                                 close_available_at="2026-01-03T00:00:00Z")
    options = dict(probability_col="p", policy="P0", tax_rate=0.0,
                   min_ev=0.05, qualification="raw")
    before = policies.select_policy_bets(frame, **options)
    after = policies.select_policy_bets(changed_close, **options)
    assert before[["bet_id", "side", "odds"]].to_dict("records") == after[
        ["bet_id", "side", "odds"]
    ].to_dict("records")


def test_portfolio_reserves_overlapping_stakes_until_label_release():
    bets = pd.DataFrame([
        {"bet_id": "a", "decision_at": "2026-01-01T10:00:00Z",
         "settle_at": "2026-01-01T12:00:00Z", "odds": 2.0, "won": True},
        {"bet_id": "b", "decision_at": "2026-01-01T11:00:00Z",
         "settle_at": "2026-01-01T13:00:00Z", "odds": 2.0, "won": False},
        {"bet_id": "c", "decision_at": "2026-01-01T12:30:00Z",
         "settle_at": "2026-01-01T14:00:00Z", "odds": 2.0, "won": False},
    ])
    ledger, summary = policies.simulate_portfolio(
        bets, strategy="flat", initial_bankroll=100.0, flat_stake=60.0,
        tax_rate=0.0,
    )
    a, b, c = ledger.set_index("bet_id").loc[["a", "b", "c"]].to_dict("records")
    assert a["placed"] is True
    assert a["free_capital_before"] == pytest.approx(100.0)
    assert a["free_capital_after"] == pytest.approx(40.0)
    assert b["placed"] is False
    assert b["free_capital_before"] == pytest.approx(40.0)
    assert c["placed"] is True
    assert c["free_capital_before"] == pytest.approx(160.0)
    assert a["decision_at"] < a["settle_at"]
    assert {"bet_id", "stake", "pnl", "decision_at", "settle_at",
            "free_capital_before", "free_capital_after", "placed", "status"}.issubset(ledger)
    assert all(math.isfinite(float(value)) for value in summary.values()
               if isinstance(value, (int, float)))


def test_portfolio_stake_is_decision_time_only_and_repeated_runs_are_identical():
    bets = pd.DataFrame([{
        "bet_id": "fixed", "decision_at": "2026-01-01T10:00:00Z",
        "settle_at": "2026-01-01T12:00:00Z", "odds": 2.5,
        "p": 0.60, "won": True,
    }])
    kwargs = dict(strategy="flat", initial_bankroll=1000.0, flat_stake=100.0,
                  tax_rate=0.12)
    ledger_a, summary_a = policies.simulate_portfolio(bets, **kwargs)
    ledger_b, summary_b = policies.simulate_portfolio(bets.assign(won=False), **kwargs)
    ledger_repeat, summary_repeat = policies.simulate_portfolio(bets, **kwargs)
    assert ledger_a.loc[0, "stake"] == ledger_b.loc[0, "stake"] == 100.0
    assert ledger_a.loc[0, "pnl"] == pytest.approx(100.0 * (2.5 * 0.88 - 1.0))
    assert ledger_b.loc[0, "pnl"] == pytest.approx(-100.0)
    pd.testing.assert_frame_equal(ledger_a, ledger_repeat)
    assert summary_a == summary_repeat
    assert summary_a != summary_b


def test_quarter_kelly_reserves_overlapping_bankroll_and_ignores_unsettled_outcome():
    bets = pd.DataFrame([
        {"bet_id": "first", "decision_at": "2026-01-01T10:00:00Z",
         "settle_at": "2026-01-01T12:00:00Z", "odds": 2.0,
         "p": 0.60, "won": True},
        {"bet_id": "second", "decision_at": "2026-01-01T11:00:00Z",
         "settle_at": "2026-01-01T13:00:00Z", "odds": 2.0,
         "p": 0.60, "won": False},
    ])
    options = dict(strategy="quarter_kelly", initial_bankroll=1000.0,
                   tax_rate=0.0, kelly_fraction=0.25, max_fraction=0.025)
    ledger, _ = policies.simulate_portfolio(bets, **options)
    flipped, _ = policies.simulate_portfolio(bets.assign(won=[False, False]), **options)
    assert ledger["stake"].tolist() == pytest.approx([25.00, 24.38])
    assert flipped.loc[1, "stake"] == pytest.approx(ledger.loc[1, "stake"])
    assert ledger.loc[0, "decision_at"] < ledger.loc[1, "decision_at"] < ledger.loc[0, "settle_at"]


def test_empty_portfolio_has_none_roi_and_duplicate_ids_are_rejected():
    empty, summary = policies.simulate_portfolio(
        pd.DataFrame(columns=["bet_id", "decision_at", "settle_at", "odds", "won"]),
        strategy="flat", initial_bankroll=100.0, flat_stake=10.0,
    )
    assert summary["roi"] is None
    assert summary["total_staked"] == 0.0
    assert summary["total_pnl"] == 0.0
    duplicate = pd.DataFrame([
        {"bet_id": "same", "decision_at": "2026-01-01T10:00:00Z",
         "settle_at": "2026-01-01T12:00:00Z", "odds": 2.0, "won": True},
        {"bet_id": "same", "decision_at": "2026-01-01T11:00:00Z",
         "settle_at": "2026-01-01T13:00:00Z", "odds": 2.0, "won": False},
    ])
    with pytest.raises(ValueError, match="duplicate.*bet_id|bet_id.*unique"):
        policies.simulate_portfolio(duplicate, strategy="flat", initial_bankroll=100.0)


def test_p5_selection_uses_only_caller_supplied_prior_cal_sigma_by_variant_year():
    frame = pd.DataFrame([
        opportunity(bet_id="seed-bet", row_id="seed", sigma_z=0.35,
                    variant="seed_only", model_year=2025),
        opportunity(bet_id="bootstrap-bet", row_id="bootstrap", sigma_z=0.35,
                    variant="block_bootstrap", model_year=2025),
    ])
    cal = pd.DataFrame([
        {"row_id": f"seed-cal-{index}", "variant": "seed_only",
         "model_year": 2025, "sigma_z": value}
        for index, value in enumerate((0.10, 0.20, 0.30, 0.40))
    ] + [
        {"row_id": f"bootstrap-cal-{index}", "variant": "block_bootstrap",
         "model_year": 2025, "sigma_z": value}
        for index, value in enumerate((0.30, 0.40, 0.50, 0.60))
    ])

    selected = policies.select_policy_bets(
        frame, policy="P5", tax_rate=0.0, min_ev=0.05,
        qualification="raw", cal_thresholds=cal,
    )

    assert selected["bet_id"].tolist() == ["bootstrap-bet"]
    assert selected.loc[0, "p5_threshold"] == pytest.approx(0.525)
    assert selected.loc[0, "p"] == pytest.approx(0.60)
    assert selected.loc[0, "p_for_ev"] == pytest.approx(0.60)
    assert selected.loc[0, "p5_threshold_source"] == "caller_prior_CAL"
    assert selected.loc[0, "p5_threshold_source_ids"] == [
        "bootstrap-cal-0", "bootstrap-cal-1", "bootstrap-cal-2", "bootstrap-cal-3"
    ]


def test_risk_side_bounds_change_ev_without_replacing_the_coherent_mean():
    selected = policies.select_policy_bets(
        pd.DataFrame([opportunity(p_low_a=0.56, p_low_b=0.0,
                                  odds_a=2.2, odds_b=1.01)]),
        policy="P0", tax_rate=0.0, min_ev=0.05, qualification="raw",
    )

    assert selected["side"].tolist() == ["A"]
    assert selected.loc[0, "p"] == pytest.approx(0.60)
    assert selected.loc[0, "p_for_ev"] == pytest.approx(0.56)

    abstained = policies.select_policy_bets(
        pd.DataFrame([opportunity(allow_a=False, odds_a=2.2, odds_b=1.01)]),
        policy="P0", tax_rate=0.0, min_ev=0.05, qualification="raw",
    )
    assert abstained.empty


def test_settlement_release_at_decision_tie_precedes_stable_decision_order():
    bets = pd.DataFrame([
        {"bet_id": "settling", "decision_at": "2026-01-01T10:00:00Z",
         "settle_at": "2026-01-01T12:00:00Z", "odds": 2.0, "won": True},
        {"bet_id": "tied-decision", "decision_at": "2026-01-01T12:00:00Z",
         "settle_at": "2026-01-01T13:00:00Z", "odds": 2.0, "won": False},
    ])

    ledger, _ = policies.simulate_portfolio(
        bets, strategy="flat", initial_bankroll=100.0, flat_stake=60.0,
        tax_rate=0.0,
    )

    assert ledger.loc[1, "free_capital_before"] == pytest.approx(160.0)
    assert ledger.loc[1, "placed"] is True


def test_provided_fraction_uses_free_capital_without_probability_inputs():
    bets = pd.DataFrame([
        {"bet_id": "first", "action_side": "B", "action_fraction": 0.025,
         "decision_at": "2026-01-01T10:00:00Z",
         "settle_at": "2026-01-01T12:00:00Z", "odds": 2.0, "won": True},
        {"bet_id": "second", "action_side": "A", "action_fraction": 0.025,
         "decision_at": "2026-01-01T11:00:00Z",
         "settle_at": "2026-01-01T13:00:00Z", "odds": 2.0, "won": False},
        {"bet_id": "abstain", "action_side": "A", "action_fraction": 0.0,
         "decision_at": "2026-01-01T11:30:00Z",
         "settle_at": "2026-01-01T14:00:00Z", "odds": 2.0, "won": False},
    ])

    ledger, summary = policies.simulate_portfolio(
        bets, strategy="provided_fraction", initial_bankroll=1000.0,
        tax_rate=0.0,
    )

    assert ledger["stake"].tolist() == pytest.approx([25.00, 24.38, 0.00])
    assert ledger["action_side"].tolist() == ["B", "A", "A"]
    assert ledger.loc[2, "placed"] is False
    assert ledger.loc[2, "skip_reason"] == "zero_action_fraction"
    assert summary["conditional_probability_error"] is None
    assert summary["conditional_ev_error"] is None


def test_scenario_timestamps_order_replay_without_backdating_actual_decision_time():
    bets = pd.DataFrame([{
        "bet_id": "scenario",
        "action_side": "A",
        "decision_at": pd.NaT,
        "scenario_decision_at": "2026-01-01T10:00:00Z",
        "scenario_label_available_at": "2026-01-01T12:00:00Z",
        "scenario_timestamp_assumption": "prior-day release scenario",
        "odds": 2.0,
        "won": True,
    }])

    ledger, _ = policies.simulate_portfolio(
        bets, strategy="flat", initial_bankroll=100.0, flat_stake=10.0,
        tax_rate=0.0,
    )

    assert pd.isna(ledger.loc[0, "decision_at"])
    assert ledger.loc[0, "decision_effective_at"] == pd.Timestamp("2026-01-01T10:00:00Z")
    assert ledger.loc[0, "decision_time_source"] == "scenario"
    assert ledger.loc[0, "settle_at"] == pd.Timestamp("2026-01-01T12:00:00Z")


def test_scenario_decision_assumption_survives_when_label_time_is_actual():
    frame = pd.DataFrame([opportunity(
        decision_at=pd.NaT,
        scenario_decision_at="2026-01-01T10:00:00Z",
        scenario_timestamp_assumption="prior-day release scenario",
    )])
    selected = policies.select_policy_bets(
        frame, policy="P0", tax_rate=0.0, min_ev=0.05, qualification="raw",
    )

    assert selected.loc[0, "settlement_time_source"] == "actual"
    assert selected.loc[0, "scenario_timestamp_assumption"] == "prior-day release scenario"
    ledger, _ = policies.simulate_portfolio(
        selected, strategy="flat", initial_bankroll=100.0,
        flat_stake=10.0, tax_rate=0.0,
    )
    assert pd.isna(ledger.loc[0, "decision_at"])
    assert ledger.loc[0, "decision_time_source"] == "scenario"
    assert ledger.loc[0, "decision_effective_at"] == pd.Timestamp("2026-01-01T10:00:00Z")
