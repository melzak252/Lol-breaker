
import numpy as np
import pandas as pd
import pytest

from src.analysis.a0_capacity_experiment import evaluate_experiment


CELLS = ("w16_e12", "w32_e12", "w64_e12", "w16_e24", "w32_e24", "w64_e24")
MODES = ("full", "no_w20", "no_organization")


def evaluation_frame(n=30):
    y = np.arange(n) % 2
    frame = pd.DataFrame({
        "golgg_match_id": [f"m{i}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n, freq="7D"),
        "y": y,
        "best_of": np.resize([3, 5, 1], n),
        "tournament_name": np.resize(["Major Spring", "League A", "League B"], n),
        "unknown_players": np.resize([0, 1, 0], n),
        "competition_tier": np.resize(["major", "regional", "regional"], n),
        "p_c0__full": np.where(y, .65, .35),
        "p_a0__full": np.where(y, .60, .40),
    })
    for cell in CELLS:
        for mode in MODES:
            frame[f"{cell}__{mode}"] = np.where(y, .7, .3)
            frame[f"{cell}__nll__{mode}"] = .5
    return frame


def protocol(n=30):
    return {
        "run_id": "test",
        "evaluation": {"bootstrap": {"draws": 5000, "seed": 20261004}},
        "cohort": {"rows": n},
        "cells": list(CELLS),
        "modes": list(MODES),
        "annual_guards": {"a0_column": "p_a0__full", "c0_column": "p_c0__full"},
        "diagnostic_market": {"enabled": False},
    }


def test_evaluation_writes_six_cell_scores_and_exact_primary_contrasts(tmp_path):
    protocol_data = protocol()
    protocol_data["inputs"] = {"prediction_source": {"sha256": "a" * 64},
                               "native_checkpoints": {"sha256": "b" * 64}}
    result = evaluate_experiment(evaluation_frame(), protocol_data, tmp_path)

    assert set(result["metrics"]) == set(CELLS)
    assert set(result["metrics"]["w16_e12"]) == set(MODES)
    assert set(result["contrasts"]) == {
        "w64_e12-w32_e12", "w16_e12-w32_e12", "w32_e24-w32_e12"
    }
    assert result["bootstrap"]["draws"] == 5000
    assert result["bootstrap"]["seed"] == 20261004
    assert result["bootstrap"]["primary_confidence_level"] == pytest.approx(1 - .05 / 3)
    assert result["source_hashes"] == protocol_data["inputs"]
    assert result["contrasts"]["w64_e12-w32_e12"]["log_loss"]["confidence_level"] == pytest.approx(1 - .05 / 3)
    sensitivity = result["tournament_sensitivity"]
    assert sensitivity["status"] == "DESCRIPTIVE_PROXY_ONLY"
    assert sensitivity["adoption_gate"] is False
    assert sensitivity["source_proxy_block_count"] == 3
    assert sensitivity["contrasts"]["w64_e12-w32_e12"]["log_loss"]["confidence_level"] == .95
    assert "not certified tournament editions" in sensitivity["block_unit"]
    assert (tmp_path / "contrasts_sensitivity.csv").is_file()
    assert result["guards"]["w16_e12"]["full"]["best_of_3"]["comparators"]["c0"]["log_loss"]["confidence_level"] == .95
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "contrasts.csv").is_file()
    assert (tmp_path / "decision.json").is_file()


def test_candidate_adoption_gates_are_independent_and_exclude_exploratory_cells(tmp_path):
    frame = evaluation_frame()
    frame["w64_e12__full"] = np.where(frame.y.eq(1), .8, .2)
    runtime = {
        "max_control_probability_difference": 1e-7,
        "side_symmetry_max_abs_error": 1e-8,
        "empty_history_probability": .5,
    }
    result = evaluate_experiment(frame, protocol(), tmp_path, runtime_proof=runtime)

    assert result["decision"]["candidate_gates"]["w64_e12"]["adopted"] is True
    assert result["decision"]["candidate_gates"]["w16_e12"]["adopted"] is False
    assert result["decision"]["candidate_gates"]["w32_e24"]["adopted"] is False
    assert set(result["decision"]["candidate_gates"]) == {"w64_e12", "w16_e12", "w32_e24"}
    assert result["decision"]["eligible_candidates"] == ["w64_e12"]



def test_verified_teacher_advantage_does_not_override_student_contingency_failure(tmp_path):
    frame = evaluation_frame()
    frame["w64_e12__full"] = np.where(frame.y.eq(1), .8, .2)
    frame["w16_e12__full"] = np.where(frame.y.eq(1), .01, .99)
    protocol_data = protocol()
    protocol_data["distillation"] = {
        "student_cell": "w16_e12",
        "verified_teacher_advantage_candidates": ["w64_e12-w32_e12"],
    }
    result = evaluate_experiment(frame, protocol_data, tmp_path)

    distillation = result["decision"]["distillation"]
    assert distillation["trigger"]["passed"] is True
    assert distillation["eligible"] is False
    assert distillation["trigger"]["teacher_advantage_evidence"][0]["verified"] is True
    assert set(distillation["contingency"]["failed"]) == {"best_of_3", "best_of_5", "major"}


def test_unverified_teacher_advantage_boolean_does_not_trigger_distillation(tmp_path):
    frame = evaluation_frame()
    frame["w16_e12__full"] = np.where(frame.y.eq(1), .01, .99)
    protocol_data = protocol()
    protocol_data["distillation"] = {
        "student_cell": "w16_e12", "verified_teacher_advantage": True,
    }
    result = evaluate_experiment(frame, protocol_data, tmp_path)

    distillation = result["decision"]["distillation"]
    assert distillation["trigger"]["passed"] is False
    assert distillation["eligible"] is False

def test_zero_support_guard_cannot_pass_even_when_supported_guards_are_good(tmp_path):
    frame = evaluation_frame()
    frame["best_of"] = 1
    result = evaluate_experiment(frame, protocol(), tmp_path)

    for cell in CELLS:
        assert result["guards"][cell]["full"]["best_of_3"]["n"] == 0
        assert result["guards"][cell]["full"]["best_of_3"]["pass"] is False
    assert result["decision"]["adopted"] is False


def test_noninferiority_uses_upper_bound_and_a0_and_c0_comparators(tmp_path):
    frame = evaluation_frame()
    frame["w16_e12__full"] = np.where(frame.y.eq(1), .01, .99)
    result = evaluate_experiment(frame, protocol(), tmp_path)

    guard = result["guards"]["w16_e12"]["full"]["best_of_3"]
    assert set(guard["comparators"]) == {"a0", "c0"}
    assert guard["pass"] is False
    assert guard["comparators"]["c0"]["log_loss"]["ci_upper"] > .005


def test_distillation_requires_student_subset_guards_and_runtime_evidence(tmp_path):
    frame = evaluation_frame()
    protocol_data = protocol()
    protocol_data["distillation"] = {
        "student_cell": "w16_e12", "kd_column": "w16_kd__full", "direct_column": "w16_direct__full"
    }
    frame["w16_kd__full"] = np.where(frame.y.eq(1), .72, .28)
    frame["w16_direct__full"] = np.where(frame.y.eq(1), .70, .30)
    result = evaluate_experiment(frame, protocol_data, tmp_path)

    decision = result["decision"]["distillation"]
    assert decision["eligible"] is True
    assert decision["runtime_evidence_status"] == "missing"
    assert decision["accuracy_brier_nonworse"] is True
    assert decision["accuracy_improvement"] is True
    assert decision["compression_only_claim"] is False


def test_zero_subset_support_is_not_a_distillation_pass(tmp_path):
    frame = evaluation_frame()
    frame["best_of"] = 1
    protocol_data = protocol()
    protocol_data["distillation"] = {
        "student_cell": "w16_e12", "kd_column": "w16_kd__full", "direct_column": "w16_direct__full"
    }
    frame["w16_kd__full"] = np.where(frame.y.eq(1), .72, .28)
    frame["w16_direct__full"] = np.where(frame.y.eq(1), .70, .30)
    result = evaluate_experiment(frame, protocol_data, tmp_path)

    assert result["decision"]["distillation"]["eligible"] is False


def test_market_decomposition_uses_fixed_side_and_complete_inventory_blocks(tmp_path):
    frame = evaluation_frame(1615)
    frame["date"] = pd.date_range("2026-01-01", periods=1615, freq="4h")
    accepted = frame[["golgg_match_id", "date", "y"]].copy()
    accepted["q"] = .45
    selected = accepted.iloc[:330].copy().reset_index(drop=True)
    selected["row_id"] = [f"pinnacle:{match_id}" for match_id in selected.golgg_match_id]
    selected["fixture_id"] = [f"fixture-{i}" for i in range(len(selected))]
    selected["bet_id"] = [f"bet-{i}" for i in range(len(selected))]
    selected["source"] = "pinnacle"
    selected["model_key"] = "new_connector"
    selected["selection_id"] = "pinnacle__new_connector__raw_ev_gt_5"
    selected["policy_name"] = "raw_ev_gt_5"
    selected["side"] = np.where(np.arange(len(selected)) % 2 == 0, "A", "B")
    selected["odds"] = 2.4
    selected["won"] = np.where(selected.side.eq("A"), selected.y, 1 - selected.y)
    selected["p_for_ev"] = .8
    selected.loc[:75, "p_for_ev"] = 1.08 / (2.4 * .88)
    selected["ev"] = .88 * selected.odds * selected.p_for_ev - 1
    selected["tax_rate"] = .12
    inventory = pd.DataFrame({
        "row_id": [f"pinnacle:m{i}" for i in range(1619)],
        "date": pd.date_range("2026-01-01", periods=1619, freq="4h"),
        "source": "pinnacle",
    })
    protocol_data = protocol(1615)
    protocol_data["diagnostic_market"] = {
        "enabled": True, "accepted_rows": accepted.to_dict("records"),
        "selection_rows": selected.to_dict("records"),
        "inventory_rows": inventory.to_dict("records"),
    }
    result = evaluate_experiment(frame, protocol_data, tmp_path)

    row = next(
        item for item in result["market_diagnostic"]["fixed_selection_rows"]
        if item["golgg_match_id"] == "m0" and item["model"] == "w16_e12" and item["mode"] == "full"
    )
    assert row["model_probability_chosen_side"] == pytest.approx(.3)
    assert row["q_chosen_side"] == pytest.approx(.45)
    assert row["term_model_market"] == pytest.approx(.88 * 2.4 * (.3 - .45))
    assert row["term_market_residual"] == pytest.approx(.88 * 2.4 * (.45 - 0))
    assert row["ev_error"] == pytest.approx(row["estimated_ev"] - row["realized_return"])
    assert len(result["market_diagnostic"]["fixed_selection_rows"]) == 330 * len(MODES) * len(CELLS)
    assert result["market_diagnostic"]["by_model"]["w16_e12"]["full"]["selected"]["n"] == 330
    assert result["market_diagnostic"]["by_model"]["w16_e12"]["full"]["selected_near_5_10"]["n"] == 76
    assert result["market_diagnostic"]["full_inventory_block_universe_rows"] == 1619
    assert result["market_diagnostic"]["policy_tuned"] is False


def test_shared_paired_bootstrap_rejects_misaligned_blocks():
    from src.analysis.production_market_benchmark import paired_cluster_interval

    with pytest.raises(ValueError, match="align"):
        paired_cluster_interval([.1, -.2], ["2026-01"])
