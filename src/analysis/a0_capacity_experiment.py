"""Evaluation and decision gates for the frozen native-capacity experiment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import date, datetime

import numpy as np
import pandas as pd
from src.analysis.bootstrap import block_ratio_interval
from src.analysis.production_market_benchmark import paired_cluster_interval


CELLS = ("w16_e12", "w32_e12", "w64_e12", "w16_e24", "w32_e24", "w64_e24")
MODES = ("full", "no_w20", "no_organization")
PRIMARY_CONTRASTS = (
    ("w64_e12-w32_e12", "w64_e12", "w32_e12"),
    ("w16_e12-w32_e12", "w16_e12", "w32_e12"),
    ("w32_e24-w32_e12", "w32_e24", "w32_e12"),
)
DRAWS = 5000
SEED = 20261004
ALPHA = 0.05 / 3


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    return value


def _probability(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
        raise ValueError(f"invalid strict probability in {column}")
    return values


def _log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def _score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    ll = _log_loss(y, p)
    return {"n": int(len(y)), "log_loss": float(ll.mean()),
            "brier": float(np.square(p - y).mean())}


def _paired_intervals(part: pd.DataFrame, delta: np.ndarray, draws: int, seed: int,
                      alpha: float = 0.05) -> dict[str, Any]:
    months = pd.to_datetime(part.date, errors="raise").dt.to_period("M").astype(str)
    interval = paired_cluster_interval(
        delta, months, draws=draws, seed=seed, confidence_level=1 - alpha,
    )
    return {
        "n": interval["n"], "blocks": interval["blocks"], "delta": interval["delta"],
        "ci_lower": interval["ci_lower"], "ci_upper": interval["ci_upper"],
        "equal_month_delta": interval["equal_block_delta"],
        "equal_month_ci_lower": interval["equal_block_ci_lower"],
        "equal_month_ci_upper": interval["equal_block_ci_upper"],
        "draws": interval["draws"], "seed": interval["seed"],
        "confidence_level": 1 - alpha, "zero_support_replicates": 0,
    }


def _tournament_sensitivity(frame: pd.DataFrame, contrasts: dict,
                            probabilities: dict, draws: int, seed: int) -> dict[str, Any]:
    if "tournament_name" not in frame:
        raise ValueError("tournament-name/year sensitivity requires tournament_name")
    names = frame.tournament_name.astype("string").fillna("unknown").astype(str)
    years = frame.date.dt.year.astype(str)
    blocks = (names + "|" + years).to_numpy()
    block_counts = pd.Series(blocks).value_counts(sort=False).sort_index()
    result = {
        "status": "DESCRIPTIVE_PROXY_ONLY",
        "block_unit": "tournament_name × calendar_year proxy; not certified tournament editions",
        "multiplicity_adjusted": False,
        "adoption_gate": False,
        "missing_tournament_name_rows": int(frame.tournament_name.isna().sum()),
        "source_proxy_block_count": int(block_counts.size),
        "source_proxy_block_counts": [
            {"block": str(block), "n": int(count)} for block, count in block_counts.items()
        ],
        "contrasts": {},
    }
    y = frame.y.to_numpy(int)
    for name, item in contrasts.items():
        candidate, reference = item["candidate"], item["reference"]
        candidate_p = probabilities[candidate, "full"]
        reference_p = probabilities[reference, "full"]
        ll = _paired_cluster_summary(
            _log_loss(y, candidate_p) - _log_loss(y, reference_p), blocks,
            draws, seed,
        )
        brier = _paired_cluster_summary(
            np.square(candidate_p - y) - np.square(reference_p - y), blocks,
            draws, seed,
        )
        result["contrasts"][name] = {"log_loss": ll, "brier": brier}
    return result


def _paired_cluster_summary(delta: np.ndarray, blocks: np.ndarray,
                            draws: int, seed: int) -> dict[str, Any]:
    interval = paired_cluster_interval(
        delta, blocks, draws=draws, seed=seed, confidence_level=.95,
    )
    return {
        "n": interval["n"], "blocks": interval["blocks"],
        "delta": interval["delta"], "ci_lower": interval["ci_lower"],
        "ci_upper": interval["ci_upper"],
        "equal_proxy_delta": interval["equal_block_delta"],
        "equal_proxy_ci_lower": interval["equal_block_ci_lower"],
        "equal_proxy_ci_upper": interval["equal_block_ci_upper"],
        "confidence_level": .95, "draws": draws, "seed": seed,
    }


def _subset_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    if "best_of" not in frame:
        raise ValueError("missing required subset field best_of")
    masks = {
        "best_of_3": pd.to_numeric(frame.best_of, errors="coerce").eq(3).to_numpy(),
        "best_of_5": pd.to_numeric(frame.best_of, errors="coerce").eq(5).to_numpy(),
    }
    if "competition_tier" in frame:
        masks["major"] = frame.competition_tier.astype("string").str.casefold().eq("major").fillna(False).to_numpy()
    else:
        raise ValueError("missing required competition_tier taxonomy for major subset")
    if "unknown_players" in frame:
        masks["source_unseen"] = pd.to_numeric(frame.unknown_players, errors="coerce").gt(0).fillna(False).to_numpy()
    elif "source_unseen" in frame:
        masks["source_unseen"] = frame.source_unseen.fillna(False).astype(bool).to_numpy()
    else:
        raise ValueError("missing source-unseen-player subset field")
    return masks


def _guard_for(part: pd.DataFrame, candidate: np.ndarray, comparator: np.ndarray,
               draws: int, seed: int) -> dict[str, Any]:
    if part.empty:
        return {"n": 0, "pass": False, "log_loss": None, "threshold": 0.005}
    y = part.y.to_numpy(dtype=int)
    delta = _log_loss(y, candidate) - _log_loss(y, comparator)
    interval = _paired_intervals(part, delta, draws, seed, alpha=.05)
    return {"n": len(part), "log_loss": interval, "threshold": 0.005,
            "pass": interval["ci_upper"] is not None and interval["ci_upper"] <= 0.005}


def _distillation(frame: pd.DataFrame, protocol: dict, guards: dict,
                 contrasts: dict, candidate_decisions: dict,
                 runtime_proof: dict | None, draws: int, seed: int) -> dict[str, Any]:
    spec = protocol.get("distillation")
    if not spec:
        return {"eligible": False, "reason": "distillation_not_in_protocol"}
    student = spec.get("student_cell", "w16_e12")
    overall = _guard_for(
        frame, _probability(frame, f"{student}__full"), _probability(frame, "p_c0__full"),
        draws, seed,
    )
    allowed_contrasts = {name for name, _, _ in PRIMARY_CONTRASTS}
    teacher_advantage_evidence = []
    for contrast_name in spec.get("verified_teacher_advantage_candidates", []):
        if contrast_name not in allowed_contrasts:
            raise ValueError(f"invalid frozen teacher-advantage contrast: {contrast_name}")
        contrast = contrasts[contrast_name]
        candidate_cell = contrast["candidate"]
        sports_guards_pass = candidate_decisions.get(candidate_cell, {}).get(
            "sports_guards_pass", False
        )
        teacher_advantage_evidence.append({
            "contrast": contrast_name, "candidate_cell": candidate_cell,
            "primary_contrast_pass": contrast["pass"],
            "sports_guards_pass": sports_guards_pass,
            "verified": bool(contrast["pass"] and sports_guards_pass),
        })
    teacher_advantage = any(item["verified"] for item in teacher_advantage_evidence)
    trigger = bool(overall["pass"] or teacher_advantage)
    required_subsets = ("best_of_3", "best_of_5", "major")
    c0_subset_guards = {
        name: guards.get(student, {}).get("full", {}).get(name, {}).get("comparators", {}).get("c0", {})
        for name in required_subsets
    }
    failed = [name for name, item in c0_subset_guards.items() if not item.get("pass", False)]
    eligible = bool(trigger and not failed)
    result = {
        "eligible": eligible,
        "reason": ("trigger_and_contingency_passed" if eligible else
                   "student_c0_subset_guard_failed" if failed else "no_distillation_trigger"),
        "trigger": {
            "width16_overall_c0_noninferiority": overall,
            "verified_width32_or_width64_teacher_advantage": teacher_advantage,
            "teacher_advantage_evidence": teacher_advantage_evidence,
            "passed": trigger,
        },
        "contingency": {
            "comparator": "c0", "required_subsets": list(required_subsets),
            "guards": c0_subset_guards, "failed": failed, "passed": not failed,
        },
    }
    if not eligible:
        return result
    kd_column, direct_column = spec.get("kd_column"), spec.get("direct_column")
    teacher_column = spec.get("teacher_column", "w32_e12__full")
    if not kd_column or kd_column not in frame or not direct_column or direct_column not in frame:
        result["result_status"] = "ELIGIBLE_NOT_EVALUATED_MISSING_KD_OR_DIRECT_PREDICTIONS"
        return result
    y = frame.y.to_numpy(dtype=int)
    kd, direct = _probability(frame, kd_column), _probability(frame, direct_column)
    ll_delta = _log_loss(y, kd) - _log_loss(y, direct)
    brier_delta = np.square(kd - y) - np.square(direct - y)
    accuracy_interval = _paired_intervals(frame, ll_delta, draws, seed, alpha=.05)
    brier_point_nonworse = bool(brier_delta.mean() <= 0)
    result["accuracy_comparison_kd_minus_direct"] = accuracy_interval
    result["accuracy_brier_delta"] = float(brier_delta.mean())
    result["accuracy_brier_nonworse"] = brier_point_nonworse
    result["accuracy_improvement"] = bool(
        accuracy_interval["delta"] is not None and accuracy_interval["delta"] < 0
        and accuracy_interval["ci_upper"] is not None and accuracy_interval["ci_upper"] < 0
        and brier_point_nonworse
    )
    masks = _subset_masks(frame)
    comparator_values = {"c0": _probability(frame, "p_c0__full")}
    if teacher_column in frame:
        comparator_values["teacher_w32_e12"] = _probability(frame, teacher_column)
    kd_guards = {}
    for comparator, values in comparator_values.items():
        overall_guard = _guard_for(frame, kd, values, draws, seed)
        subset_guards = {
            subset: _guard_for(frame.loc[mask].reset_index(drop=True), kd[mask], values[mask], draws, seed)
            for subset, mask in masks.items()
        }
        kd_guards[comparator] = {
            "overall": overall_guard, "subsets": subset_guards,
            "pass": bool(overall_guard["pass"] and all(item["pass"] for item in subset_guards.values())),
        }
    kd_noninferior = bool(kd_guards and all(item["pass"] for item in kd_guards.values()))
    result["kd_sports_guards"] = kd_guards
    result["kd_noninferiority"] = kd_noninferior
    runtime = runtime_proof or {}
    runtime_fields = ("teacher_latency_ms", "kd_latency_ms",
                      "teacher_peak_memory_bytes", "kd_peak_memory_bytes")
    runtime_complete = all(key in runtime for key in runtime_fields)
    latency_reduction = bool(runtime_complete and float(runtime["kd_latency_ms"]) <
                             float(runtime["teacher_latency_ms"]))
    memory_reduction = bool(runtime_complete and int(runtime["kd_peak_memory_bytes"]) <
                            int(runtime["teacher_peak_memory_bytes"]))
    result["result_status"] = "EVALUATED"
    result["runtime_evidence_status"] = "verified" if runtime_complete else "missing"
    result["compression_evidence"] = {
        "comparison_model": "teacher_w32_e12",
        "latency_reduction": latency_reduction, "memory_reduction": memory_reduction,
        **({key: runtime[key] for key in runtime_fields} if runtime_complete else {}),
    }
    result["compression_only_claim"] = bool(kd_noninferior and latency_reduction and memory_reduction)
    return result


def _market_diagnostic(frame: pd.DataFrame, protocol: dict, draws: int, seed: int) -> dict[str, Any] | None:
    spec = protocol.get("diagnostic_market", {})
    if not spec.get("enabled", False):
        return None
    accepted_rows = spec.get("accepted_rows")
    selection_rows = spec.get("selection_rows")
    inventory_rows = spec.get("inventory_rows")
    if accepted_rows is None or selection_rows is None or inventory_rows is None:
        raise ValueError("enabled Pinnacle audit requires frozen accepted, selection, and inventory rows")
    accepted = pd.DataFrame(accepted_rows)
    selected = pd.DataFrame(selection_rows)
    inventory = pd.DataFrame(inventory_rows)
    if len(accepted) != 1615 or len(inventory) != 1619:
        raise ValueError("frozen Pinnacle diagnostic cohorts must retain 1615 accepted and 1619 inventory rows")
    accepted_required = {"golgg_match_id", "date", "y", "q"}
    selection_required = {
        "row_id", "fixture_id", "golgg_match_id", "date", "bet_id", "selection_id",
        "source", "model_key", "policy_name", "side", "odds", "p_for_ev", "won", "ev", "tax_rate",
    }
    inventory_required = {"row_id", "date"}
    for label, table, required in (
        ("accepted", accepted, accepted_required),
        ("selection", selected, selection_required),
        ("inventory", inventory, inventory_required),
    ):
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"frozen {label} market rows missing columns: {missing}")
    if accepted.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("accepted Pinnacle cohort has duplicate fixture identities")
    if "source" in accepted and not accepted.source.astype(str).eq("pinnacle").all():
        raise ValueError("accepted diagnostic cohort contains non-Pinnacle sources")
    if inventory.row_id.astype(str).duplicated().any():
        raise ValueError("Pinnacle full inventory has duplicate row identities")
    if "source" in inventory and not inventory.source.astype(str).eq("pinnacle").all():
        raise ValueError("full diagnostic inventory contains non-Pinnacle rows")
    if "source" in selected and not selected.source.astype(str).eq("pinnacle").all():
        raise ValueError("fixed selection ledger contains non-Pinnacle rows")
    if "model_key" in selected and not selected.model_key.astype(str).eq("new_connector").all():
        raise ValueError("fixed selection ledger is not the saved new-connector policy")
    if "policy_name" in selected and not selected.policy_name.astype(str).eq("raw_ev_gt_5").all():
        raise ValueError("fixed selection ledger has changed policy membership")
    if selected.row_id.astype(str).duplicated().any():
        # The near-band membership may be repeated as a separate source group;
        # duplicates must be identical saved bets and are collapsed by row ID.
        comparison = [column for column in selection_required if column in selected]
        for _, duplicate_group in selected.groupby(selected.row_id.astype(str), sort=False):
            if len(duplicate_group) > 1 and any(
                duplicate_group[column].nunique(dropna=False) != 1
                for column in comparison
            ):
                raise ValueError("duplicated fixed selection rows disagree on saved identity or wager")
    target_raw = int(spec.get("raw_ev_selected_count", 330))
    target_near = int(spec.get("near_band_count", 76))
    selected = selected.sort_values("row_id", kind="stable").drop_duplicates("row_id", keep="first").copy()
    if len(selected) != target_raw:
        raise ValueError(f"saved raw-EV selection changed: expected {target_raw}, got {len(selected)}")
    saved_ev = pd.to_numeric(selected.ev, errors="coerce").to_numpy(float)
    if not np.isfinite(saved_ev).all() or np.any(saved_ev <= .05):
        raise ValueError("fixed selection ledger contains wagers outside raw EV >5%")
    near_mask = np.isfinite(saved_ev) & (saved_ev > .05) & (saved_ev <= .10)
    if "near_band" in selected:
        supplied_near = selected.near_band.fillna(False).astype(bool).to_numpy()
        if not np.array_equal(supplied_near, near_mask):
            raise ValueError("near-band membership differs from frozen saved EV")
    if int(near_mask.sum()) != target_near:
        raise ValueError(f"saved near-band membership changed: expected {target_near}, got {int(near_mask.sum())}")
    if not selected.side.astype(str).str.upper().isin(["A", "B"]).all():
        raise ValueError("fixed selection has an invalid chosen side")
    if not selected.golgg_match_id.astype(str).isin(accepted.golgg_match_id.astype(str)).all():
        raise ValueError("fixed selection fixture is outside the accepted Pinnacle cohort")
    accepted = accepted.copy()
    accepted["golgg_match_id"] = accepted.golgg_match_id.astype(str)
    accepted["_date"] = pd.to_datetime(accepted.date, errors="raise").dt.date
    selected = selected.copy()
    selected["golgg_match_id"] = selected.golgg_match_id.astype(str)
    selected["_date"] = pd.to_datetime(selected.date, errors="raise").dt.date
    inventory = inventory.copy()
    inventory["row_id"] = inventory.row_id.astype(str)
    inventory["_date"] = pd.to_datetime(inventory.date, errors="raise").dt.date
    accepted_by_id = accepted.set_index("golgg_match_id")
    selected_expected_date = selected.golgg_match_id.map(accepted_by_id["_date"])
    if not np.array_equal(selected._date.to_numpy(), selected_expected_date.to_numpy()):
        raise ValueError("saved selection date differs from accepted Pinnacle source")
    if "fixture_id" in accepted_by_id:
        source_fixture = selected.golgg_match_id.map(accepted_by_id.fixture_id).astype(str).to_numpy()
        if not np.array_equal(selected.fixture_id.astype(str).to_numpy(), source_fixture):
            raise ValueError("saved selection fixture ID differs from accepted Pinnacle source")
    prediction = frame.assign(golgg_match_id=frame.golgg_match_id.astype(str)).set_index("golgg_match_id")
    accepted_index = accepted.set_index("golgg_match_id")
    assessment = accepted.loc[accepted.golgg_match_id.isin(prediction.index)].copy()
    if len(assessment) != 1615:
        raise ValueError("accepted Pinnacle assessment rows do not all align to native predictions")
    if not np.array_equal(assessment.y.to_numpy(int),
                          prediction.loc[assessment.golgg_match_id].y.to_numpy(int)):
        raise ValueError("accepted Pinnacle outcomes disagree with native assessment outcomes")
    aligned_market_predictions = prediction.loc[accepted.golgg_match_id]
    aligned_selection_predictions = prediction.loc[selected.golgg_match_id]
    q = pd.to_numeric(accepted.q, errors="coerce").to_numpy(float)
    y = pd.to_numeric(accepted.y, errors="coerce").to_numpy(int)
    if not np.isfinite(q).all() or np.any((q <= 0) | (q >= 1)):
        raise ValueError("invalid no-vig Pinnacle market probability")
    market_scores = {}
    for cell in CELLS:
        market_scores[cell] = {}
        for mode in MODES:
            p = _probability(aligned_market_predictions, f"{cell}__{mode}")
            market_scores[cell][mode] = _score(y, p)
    market_scores["market_reference"] = _score(y, q)
    accepted_index = accepted.set_index("golgg_match_id")
    selected_q_a = pd.to_numeric(
        selected.golgg_match_id.map(accepted_index.q), errors="coerce"
    ).to_numpy(float)
    selected_y_a = pd.to_numeric(
        selected.golgg_match_id.map(accepted_index.y), errors="coerce"
    ).to_numpy(int)
    is_side_a = selected.side.astype(str).str.upper().eq("A").to_numpy()
    selected_q = np.where(is_side_a, selected_q_a, 1 - selected_q_a)
    selected_won = pd.to_numeric(selected.won, errors="coerce").to_numpy(float)
    y_side = np.where(is_side_a, selected_y_a, 1 - selected_y_a)
    if not np.isin(selected_won, [0, 1]).all() or not np.array_equal(selected_won.astype(int), y_side):
        raise ValueError("saved chosen-side outcome disagrees with canonical Pinnacle outcome")
    if "y" in selected:
        saved_y = pd.to_numeric(selected.y, errors="coerce").to_numpy(int)
        if not np.array_equal(saved_y, selected_y_a):
            raise ValueError("saved selection canonical outcome differs from accepted Pinnacle source")
    if "q" in selected:
        saved_q_a = pd.to_numeric(selected.q, errors="coerce").to_numpy(float)
        if not np.allclose(saved_q_a, selected_q_a, atol=1e-12, rtol=0):
            raise ValueError("saved selection market probability disagrees with frozen accepted quote")
    odds = pd.to_numeric(selected.odds, errors="coerce").to_numpy(float)
    saved_p = pd.to_numeric(selected.p_for_ev, errors="coerce").to_numpy(float)
    saved_ev = pd.to_numeric(selected.ev, errors="coerce").to_numpy(float)
    if (not np.isfinite(odds).all() or np.any(odds <= 1) or not np.isfinite(saved_p).all()
            or np.any((saved_p <= 0) | (saved_p >= 1)) or not np.isfinite(saved_ev).all()):
        raise ValueError("invalid saved chosen-side odds, probability, or EV")
    odds_a = (pd.to_numeric(selected.golgg_match_id.map(accepted_index.odds_a), errors="coerce").to_numpy(float)
              if "odds_a" in accepted_index else None)
    odds_b = (pd.to_numeric(selected.golgg_match_id.map(accepted_index.odds_b), errors="coerce").to_numpy(float)
              if "odds_b" in accepted_index else None)
    if odds_a is not None and odds_b is not None:
        source_odds = np.where(is_side_a, odds_a, odds_b)
        if not np.allclose(odds, source_odds, atol=1e-12, rtol=0):
            raise ValueError("saved chosen-side odds differ from accepted Pinnacle source")
    tax_values = (pd.to_numeric(selected.tax_rate, errors="coerce").to_numpy(float)
                  if "tax_rate" in selected else np.full(len(selected), .12))
    if not np.allclose(tax_values, .12, atol=0, rtol=0):
        raise ValueError("saved selected bets do not use the frozen 12% tax")
    if not np.allclose(.88 * odds * saved_p - 1, saved_ev, atol=1e-10, rtol=0):
        raise ValueError("saved EV does not match fixed chosen-side probability and price")
    if "p_c0_connector" in accepted_index:
        saved_a = pd.to_numeric(
            selected.golgg_match_id.map(accepted_index.p_c0_connector), errors="coerce"
        ).to_numpy(float)
        source_p = np.where(is_side_a, saved_a, 1 - saved_a)
        if not np.allclose(saved_p, source_p, atol=1e-12, rtol=0):
            raise ValueError("saved selected probability differs from frozen connector probability")
    inventory_dates = pd.to_datetime(inventory.date, errors="raise")
    inventory_month = inventory_dates.dt.to_period("M").astype(str).to_numpy()
    inventory_ids = inventory.row_id.to_numpy(str)
    row_id_to_position = {key: index for index, key in enumerate(inventory_ids)}
    if not selected.row_id.astype(str).isin(row_id_to_position).all():
        raise ValueError("fixed saved selections are outside full Pinnacle inventory")
    inventory_position = selected.row_id.astype(str).map(row_id_to_position).to_numpy(int)
    if not np.array_equal(selected._date.to_numpy(), inventory.loc[inventory_position, "_date"].to_numpy()):
        raise ValueError("fixed selections disagree with source inventory dates")
    by_model = {}
    output_rows = []
    for cell in CELLS:
        by_model[cell] = {}
        for mode in MODES:
            candidate_a = _probability(aligned_selection_predictions, f"{cell}__{mode}")
            candidate_selected = np.where(is_side_a, candidate_a, 1 - candidate_a)
            estimated_ev = .88 * odds * candidate_selected - 1
            realized_return = .88 * odds * selected_won - 1
            ev_error = estimated_ev - realized_return
            model_market = .88 * odds * (candidate_selected - selected_q)
            market_residual = .88 * odds * (selected_q - selected_won)
            if not np.allclose(ev_error, model_market + market_residual, atol=1e-12, rtol=0):
                raise ArithmeticError("EV error decomposition identity failed")
            selected_records = selected.drop(columns="_date").copy()
            selected_records["model"] = cell
            selected_records["mode"] = mode
            selected_records["model_probability_chosen_side"] = candidate_selected
            selected_records["q_chosen_side"] = selected_q
            selected_records["estimated_ev"] = estimated_ev
            selected_records["realized_return"] = realized_return
            selected_records["term_model_market"] = model_market
            selected_records["term_market_residual"] = market_residual
            selected_records["ev_error"] = ev_error
            selected_records["near_band"] = near_mask
            output_rows.extend(selected_records.to_dict("records"))
            model_groups = {}
            for group_name, group_mask in (("selected", np.ones(len(selected), dtype=bool)),
                                            ("selected_near_5_10", near_mask)):
                group_metrics = {}
                for metric, values in (("ev_error", ev_error), ("model_market_component", model_market),
                                       ("market_residual_component", market_residual)):
                    numerators = np.zeros(len(inventory), dtype=float)
                    denominators = np.zeros(len(inventory), dtype=float)
                    numerators[inventory_position[group_mask]] = values[group_mask]
                    denominators[inventory_position[group_mask]] = 1.0
                    group_metrics[metric] = block_ratio_interval(
                        numerators, denominators, inventory_month, draws=draws, seed=seed,
                    )
                model_groups[group_name] = {
                    "n": int(group_mask.sum()), "metrics": group_metrics,
                    "mean_estimated_ev": float(estimated_ev[group_mask].mean()),
                    "mean_realized_return": float(realized_return[group_mask].mean()),
                }
            by_model[cell][mode] = model_groups
    return {
        "status": "COMPLETED_DIAGNOSTIC", "accepted_fixture_count": len(accepted),
        "inventory_count": len(inventory), "raw_ev_selected_count": target_raw,
        "near_band_count": target_near, "market_scores": market_scores,
        "fixed_selection_rows": output_rows, "by_model": by_model,
        "full_inventory_months": sorted(set(inventory_month)),
        "full_inventory_block_universe_rows": len(inventory),
        "policy_tuned": False, "g0_status": "BLOCKED",
    }


def _runtime_invariants(proof: dict | None, candidate: str) -> dict[str, Any]:
    proof = proof or {}
    cells = proof.get("cells", {})
    scoped = cells.get(candidate, {}) if isinstance(cells, dict) else {}
    values = {**proof, **scoped}

    def lookup(direct_keys, section_key, section_keys):
        for key in direct_keys:
            if key in values:
                return values[key]
        section = values.get(section_key, {})
        if isinstance(section, dict):
            for key in section_keys:
                if key in section:
                    return section[key]
        return None

    parity = lookup(
        ("max_control_probability_difference", "control_parity_max_abs_difference",
         "control_max_abs_probability_delta"),
        "control_parity", ("max_abs_probability_difference", "max_abs_delta"),
    )
    symmetry = lookup(
        ("side_symmetry_max_abs_error", "max_side_symmetry_error"),
        "side_symmetry", ("max_abs_error", "max_abs_delta"),
    )
    neutral = lookup(
        ("empty_history_probability", "cold_start_probability", "neutral_probability"),
        "cold_start", ("empty_history_probability", "probability"),
    )
    for name, value in (("control_parity_max_abs_difference", parity),
                        ("side_symmetry_max_abs_error", symmetry),
                        ("empty_history_probability", neutral)):
        if value is not None:
            try:
                if not np.isfinite(float(value)):
                    value = None
            except (TypeError, ValueError):
                value = None
        if name == "control_parity_max_abs_difference":
            parity = value
        elif name == "side_symmetry_max_abs_error":
            symmetry = value
        else:
            neutral = value
    checks = {
        "assessment_coverage": True,
        "control_parity": parity is not None and float(parity) <= 2e-6,
        "side_symmetry": symmetry is not None and float(symmetry) <= 1e-6,
        "cold_start_neutrality": neutral is not None and abs(float(neutral) - .5) <= 1e-4,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "control_parity_max_abs_difference": parity,
        "side_symmetry_max_abs_error": symmetry,
        "empty_history_probability": neutral,
    }


def evaluate_experiment(predictions: pd.DataFrame, protocol: dict, output_dir,
                        runtime_proof: dict | None = None) -> dict[str, Any]:
    """Score aligned saved predictions, apply frozen sports gates, and write reports."""
    frame = predictions.copy().reset_index(drop=True)
    required = {"golgg_match_id", "date", "y", "best_of", "p_c0__full", "p_a0__full"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"predictions missing required columns: {missing}")
    if frame.empty or frame.golgg_match_id.isna().any() or frame.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("assessment cohort is empty or has duplicate/missing identity")
    expected_rows = protocol.get("cohort", {}).get("rows", protocol.get("split", {}).get("assessment_rows"))
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise ValueError(f"assessment row count changed: expected {expected_rows}, got {len(frame)}")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame.date.isna().any():
        raise ValueError("assessment rows require a valid calendar date")
    y = pd.to_numeric(frame.y, errors="coerce").to_numpy(float)
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid assessment outcomes")
    frame["y"] = y.astype(int)
    draws = int(protocol.get("evaluation", {}).get("bootstrap", {}).get("draws", DRAWS))
    seed = int(protocol.get("evaluation", {}).get("bootstrap", {}).get("seed", SEED))
    if draws < DRAWS or seed != SEED:
        raise ValueError("capacity evaluation requires at least 5000 resamples with seed 20261004")

    metrics: dict[str, dict[str, Any]] = {}
    probabilities: dict[tuple[str, str], np.ndarray] = {}
    for cell in CELLS:
        metrics[cell] = {}
        for mode in MODES:
            probability_column = f"{cell}__{mode}"
            nll_column = f"{cell}__nll__{mode}"
            if probability_column not in frame or nll_column not in frame:
                raise ValueError(f"missing cell output columns: {probability_column}, {nll_column}")
            p = _probability(frame, probability_column)
            probabilities[cell, mode] = p
            score = _score(frame.y.to_numpy(int), p)
            monthly = frame.assign(_month=pd.to_datetime(frame.date).dt.to_period("M").astype(str))
            month_ll, month_brier = [], []
            for _, month in monthly.groupby("_month", sort=True):
                month_y = month.y.to_numpy(int)
                month_p = p[month.index.to_numpy()]
                month_ll.append(float(_log_loss(month_y, month_p).mean()))
                month_brier.append(float(np.square(month_p - month_y).mean()))
            score["equal_month_log_loss"] = float(np.mean(month_ll))
            score["equal_month_brier"] = float(np.mean(month_brier))
            nll = pd.to_numeric(frame[nll_column], errors="coerce").to_numpy(float)
            if not np.isfinite(nll).all() or np.any(nll < 0):
                raise ValueError(f"invalid native scoreline NLL in {nll_column}")
            score["scoreline_nll"] = float(nll.mean())
            score["scoreline_nll_n"] = int(len(nll))
            raw_nll_column = f"{cell}__raw_nll__{mode}"
            if raw_nll_column in frame:
                raw_nll = pd.to_numeric(frame[raw_nll_column], errors="coerce").to_numpy(float)
                if not np.isfinite(raw_nll).all() or np.any(raw_nll < 0):
                    raise ValueError(f"invalid raw native scoreline NLL in {raw_nll_column}")
                score["raw_scoreline_nll"] = float(raw_nll.mean())
            metrics[cell][mode] = score

    guards: dict[str, Any] = {}
    masks = _subset_masks(frame)
    for cell in CELLS:
        guards[cell] = {}
        for mode in MODES:
            candidate_values = probabilities[cell, mode]
            c0_column, a0_column = f"p_c0__{mode}", f"p_a0__{mode}"
            if c0_column not in frame or a0_column not in frame:
                guards[cell][mode] = {
                    "status": "UNAVAILABLE_MATCHED_MODE_BASELINES",
                    "n": len(frame), "pass": None, "subsets": {},
                }
                continue
            comparator_values = {"c0": _probability(frame, c0_column),
                                 "a0": _probability(frame, a0_column)}
            guards[cell][mode] = {}
            for subset, mask in masks.items():
                part = frame.loc[mask].reset_index(drop=True)
                comparator_results = {
                    comparator: _guard_for(part, candidate_values[mask], values[mask], draws, seed)
                    for comparator, values in comparator_values.items()
                }
                guards[cell][mode][subset] = {
                    "n": len(part), "comparators": comparator_results,
                    "pass": bool(len(part) > 0 and all(item["pass"] for item in comparator_results.values())),
                }

    contrasts = {}
    y_int = frame.y.to_numpy(int)
    for name, candidate_cell, reference_cell in PRIMARY_CONTRASTS:
        candidate = probabilities[candidate_cell, "full"]
        reference = probabilities[reference_cell, "full"]
        ll_delta = _log_loss(y_int, candidate) - _log_loss(y_int, reference)
        brier_delta = np.square(candidate - y_int) - np.square(reference - y_int)
        ll_interval = _paired_intervals(frame, ll_delta, draws, seed, alpha=ALPHA)
        brier_interval = _paired_intervals(frame, brier_delta, draws, seed, alpha=ALPHA)
        contrasts[name] = {
            "candidate": candidate_cell, "reference": reference_cell,
            "log_loss": ll_interval, "brier": brier_interval,
            "pass": bool(ll_interval["ci_upper"] is not None and ll_interval["ci_upper"] < 0
                         and brier_delta.mean() <= 0),
        }
    tournament_sensitivity = _tournament_sensitivity(
        frame, contrasts, probabilities, DRAWS, SEED
    )
    expected_coverage = expected_rows is not None and len(frame) == int(expected_rows)
    candidate_decisions = {}
    for contrast_name, item in contrasts.items():
        cell = item["candidate"]
        cell_guards = guards[cell]["full"]
        subset_guards_pass = all(cell_guards[subset]["pass"] for subset in masks)
        runtime = _runtime_invariants(runtime_proof, cell)
        adopted = bool(item["pass"] and subset_guards_pass and expected_coverage and runtime["passed"])
        candidate_decisions[cell] = {
            "primary_contrast": contrast_name, "primary_contrast_pass": item["pass"],
            "full_mode_subset_guards": cell_guards,
            "sports_guards_pass": subset_guards_pass,
            "complete_assessment_coverage": bool(expected_coverage),
            "runtime_invariants": runtime, "adopted": adopted,
        }
    eligible_candidates = [cell for cell, item in candidate_decisions.items() if item["adopted"]]
    decision = {
        "status": "ADOPTED_FOR_RESEARCH" if eligible_candidates else "NOT_ADOPTED",
        "adopted": bool(eligible_candidates), "eligible_candidates": eligible_candidates,
        "candidate_gates": candidate_decisions,
        "complete_assessment_coverage": bool(expected_coverage),
    }
    decision["distillation"] = _distillation(
        frame, protocol, guards, contrasts, candidate_decisions, runtime_proof, draws, seed
    )
    market = _market_diagnostic(frame, protocol, draws, seed)
    result = {
        "schema": "a0_capacity_evaluation_v1", "run_id": protocol.get("run_id"),
        "status": "COMPLETED_DEVELOPMENT_ONLY", "row_count": len(frame), "metrics": metrics,
        "contrasts": contrasts, "guards": guards, "decision": decision,
        "tournament_sensitivity": tournament_sensitivity,
        "bootstrap": {"draws": draws, "seed": seed,
                      "primary_confidence_level": 1 - ALPHA,
                      "guard_confidence_level": .95,
                      "interval": "paired_calendar_month_blocks",
                      "estimands": ["match_weighted", "equal_month"]},
        "market_diagnostic": market,
        "source_hashes": protocol.get("inputs", protocol.get("source_sha256", {})),
        "limitations": ["Historical development evaluation; no prospective evidence.",
                        "Betting G0 remains blocked; betting results are diagnostic only."],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(result)
    market_summary = None
    if market is not None:
        market_summary = {key: value for key, value in market.items()
                          if key != "fixed_selection_rows"}
    (output / "metrics.json").write_text(json.dumps({
        "metrics": safe["metrics"], "guards": safe["guards"],
        "bootstrap": safe["bootstrap"], "market_diagnostic": market_summary,
        "tournament_sensitivity": safe["tournament_sensitivity"],
        "source_hashes": safe["source_hashes"],
    }, indent=2, sort_keys=True) + "\n")
    rows = []
    candidate_gates = decision["candidate_gates"]
    for name, item in contrasts.items():
        row = {"contrast": name, "candidate": item["candidate"], "reference": item["reference"],
               "primary_contrast_pass": item["pass"],
               "candidate_adopted": candidate_gates[item["candidate"]]["adopted"]}
        for measure in ("log_loss", "brier"):
            for key, value in item[measure].items():
                row[f"{measure}_{key}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "contrasts.csv", index=False)
    sensitivity_rows = []
    for contrast_name, measures in tournament_sensitivity["contrasts"].items():
        for measure, interval in measures.items():
            sensitivity_rows.append({
                "contrast": contrast_name, "measure": measure,
                "status": tournament_sensitivity["status"],
                "block_unit": tournament_sensitivity["block_unit"],
                **interval,
            })
    pd.DataFrame(sensitivity_rows).to_csv(
        output / "contrasts_sensitivity.csv", index=False,
    )
    decision_market = None if market is None else {
        key: value for key, value in market.items() if key != "fixed_selection_rows"
    }
    (output / "decision.json").write_text(json.dumps({
        "decision": safe["decision"], "market_diagnostic": decision_market,
        "source_hashes": safe["source_hashes"],
    }, indent=2, sort_keys=True) + "\n")
    if market and market.get("fixed_selection_rows") is not None:
        pd.DataFrame(market["fixed_selection_rows"]).to_csv(
            output / "selected_ev_decomposition.csv", index=False,
        )
    return safe
