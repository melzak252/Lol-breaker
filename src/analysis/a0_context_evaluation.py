"""Evaluate a frozen context repair without dropping missing forecasts or refitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.a0_error_audit import ROOT, _decoded, save_json, sha256
from src.analysis.bootstrap import block_ratio_interval
from src.analysis.ev_input_repair import attach_repaired_predictions, selected_reliability, _select
from src.analysis.probability_metrics import binary_log_loss_vector, evaluate_binary_probabilities

EPSILON = 1e-12
DRAWS = 5000
SEED = 20260930


def _snapshot(protocol: dict, name: str) -> pd.DataFrame:
    receipt = json.loads((ROOT / protocol["feature_dir"] / f"{name}_receipt.json").read_text())
    path = ROOT / protocol["prediction_dir"] / f"{name}.parquet"
    if sha256(path) != receipt["output_sha256"][str(path.relative_to(ROOT))]:
        raise ValueError(f"Changed snapshot predictions: {name}")
    frame = pd.read_parquet(path)
    if frame.origin_id.duplicated().any():
        raise ValueError("Repeated origin in snapshot")
    return frame


def _attach(opportunities: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    frame = opportunities.drop(columns=[column for column in opportunities
        if column == "p_golgg" or column.startswith("repair_")]).copy()
    frame["baseline_available"] = frame.available
    frame["p_baseline"] = frame.p
    result = attach_repaired_predictions(frame, predictions.loc[predictions.cohort.isin(frame.cohort.unique())])
    result["available"] &= result.quote_contract_ok.astype(bool)
    result.loc[~result.quote_contract_ok.astype(bool), "exclusion_reason"] = "quote_state_or_binding_unresolved"
    result["context_primary_eligible"] = result.available & ~result.conflict_q25.astype(bool)
    return result


def _scores(frame: pd.DataFrame, column: str) -> dict:
    if frame.empty:
        return {"n": 0}
    p, y = frame[column].to_numpy(float), frame.y.to_numpy(float)
    if not np.isfinite(p).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("Scoring requires complete probabilities and binary outcomes")
    return {"n": len(frame), **evaluate_binary_probabilities(y, p, epsilon=EPSILON, include_accuracy=True)}


def _paired(universe: pd.DataFrame, selected: pd.Series, left: str, right: str) -> dict:
    mask = selected.to_numpy(bool)
    numerator = np.zeros(len(universe))
    rows = universe.loc[selected]
    numerator[mask] = (binary_log_loss_vector(rows.y.to_numpy(float), rows[left].to_numpy(float), epsilon=EPSILON)
                       - binary_log_loss_vector(rows.y.to_numpy(float), rows[right].to_numpy(float), epsilon=EPSILON))
    return {"monthly": block_ratio_interval(numerator, mask.astype(float), universe.month, DRAWS, SEED),
            "edition_proxy": block_ratio_interval(numerator, mask.astype(float), universe.edition_proxy.fillna("unknown"), DRAWS, SEED),
            "blocking_caveat": "Tournament-name/year proxy, not independently validated tournament editions."}


def _players(value) -> frozenset[str]:
    return frozenset(str(item) for item in _decoded(value, []) if item is not None)


def run(protocol: dict) -> dict:
    output = ROOT / protocol["report_dir"] / "context_evaluation"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite context evaluation: {output}")
    opportunities = pd.read_parquet(protocol["paths"]["audit_opportunities"])
    origins = pd.read_parquet(protocol["paths"]["audit_origins"])
    saved = pd.read_parquet(protocol["paths"]["audit_bets"])
    baseline, context = (_snapshot(protocol, name) for name in ("baseline", "context"))
    if len(opportunities) != 66040 or len(origins) != 4626:
        raise ValueError("Frozen evaluation inventory changed")
    check = _attach(opportunities, baseline)
    if not check.available.equals(opportunities.available):
        raise ValueError("Reattached baseline changed non-roster admission semantics")
    known = opportunities.p.notna()
    if not np.allclose(check.loc[known, "p"], opportunities.loc[known, "p"], atol=EPSILON, rtol=0):
        raise ValueError("Baseline opportunity probabilities differ")
    revised = _attach(opportunities, context)
    if len(revised) != len(opportunities) or not revised.row_id.equals(opportunities.row_id):
        raise ValueError("Context evaluation lost or reordered opportunity inventory")
    scope = revised.cohort.eq("oddspapi_common") & revised.source.eq("best_price") & revised.horizon.eq("T-1h")
    universe = revised.loc[scope].copy()
    primary = universe.loc[universe.audit_primary_eligible].copy()
    if len(primary) != 1852:
        raise ValueError("Frozen primary cohort changed")
    common = primary.context_primary_eligible
    new_primary = universe.loc[universe.context_primary_eligible].copy()
    available_origins = context.status.eq("available")
    baseline_origins = baseline.set_index("origin_id").status.eq("available").reindex(context.origin_id).to_numpy()
    coverage = revised.groupby(["cohort", "source", "horizon"], dropna=False).agg(
        inventory=("row_id", "size"), baseline_available=("baseline_available", "sum"), context_available=("available", "sum"),
        baseline_quality_eligible=("audit_primary_eligible", "sum"), context_quality_eligible=("context_primary_eligible", "sum")).reset_index()
    coverage["lost_available"] = revised.assign(lost=revised.baseline_available & ~revised.available).groupby(["cohort", "source", "horizon"], dropna=False).lost.sum().to_numpy()
    coverage["restored_available"] = revised.assign(restored=~revised.baseline_available & revised.available).groupby(["cohort", "source", "horizon"], dropna=False).restored.sum().to_numpy()
    forecast_fields = ["origin_id", "status", "missing_reason", "roster_a", "roster_b", "roster_source_a", "roster_source_b", "competition_context", "p_golgg"]
    changes = origins.merge(context[forecast_fields].rename(columns={key: "context_" + key for key in forecast_fields if key != "origin_id"}),
                            on="origin_id", how="left", validate="one_to_one")
    for side in ("a", "b"):
        changes[f"changed_roster_{side}"] = [
            _players(before) != _players(after)
            for before, after in zip(changes[f"roster_{side}"], changes[f"context_roster_{side}"])]
        changes[f"context_same_actual_players_{side}"] = [
            _players(after) == _players(actual) and len(_players(actual)) == 5
            for after, actual in zip(changes[f"context_roster_{side}"], changes[f"canonical_roster_{side}"])]
    changes["changed_roster"] = changes.changed_roster_a | changes.changed_roster_b
    changes["context_same_actual_players"] = changes.context_same_actual_players_a & changes.context_same_actual_players_b
    frozen = saved.loc[saved.cohort.eq("oddspapi_common") & saved.source.eq("best_price") & saved.horizon.eq("T-1h")
                       & saved.quality.eq("exclude_q25_conflicts") & saved.policy.eq("raw_ev_gt_5")].copy()
    if len(frozen) != 498 or frozen.row_id.duplicated().any():
        raise ValueError("Fixed selection inventory changed")
    frozen = frozen.merge(context[["origin_id", "status", "missing_reason", "p_golgg"]].rename(
        columns={"status": "context_status", "missing_reason": "context_missing_reason", "p_golgg": "context_p_event"}), on="origin_id", how="left", validate="one_to_one")
    frozen["context_p_selected"] = np.where(frozen.side.eq("A"), frozen.context_p_event, 1 - frozen.context_p_event)
    retained = frozen.loc[frozen.context_status.eq("available")].copy()
    fixed_before = retained.assign(p=retained.p_model_bet)
    fixed_after = retained.assign(p=retained.context_p_selected)
    fixed_stats = {"inventory": len(frozen), "available": len(retained), "withheld": len(frozen)-len(retained),
                   "missing_reasons": frozen.loc[frozen.context_status.ne("available"), "context_missing_reason"].value_counts(dropna=False).to_dict(),
                   "baseline_same_selections": selected_reliability(universe, fixed_before, tax_rate=.12),
                   "context_same_selections": selected_reliability(universe, fixed_after, tax_rate=.12)}
    contrasts = pd.DataFrame({"row_id": retained.row_id, "change": (retained.context_p_selected-retained.p_model_bet)*retained.odds.astype(float)*.88, "retained": 1.})
    aligned = universe[["row_id", "month"]].merge(contrasts, on="row_id", how="left", validate="one_to_one")
    fixed_stats["paired_ev_change"] = block_ratio_interval(aligned.change.fillna(0), aligned.retained.fillna(0), aligned.month, DRAWS, SEED)
    policies, policy_rows = {}, []
    for name in ("raw_ev_gt_5", "current_application_filters", "market_consensus_raw_ev_gt_5"):
        selected = _select(new_primary, name)
        policies[name] = selected_reliability(universe, selected, tax_rate=.12)
        policies[name]["near_5_to_10_percent"] = selected_reliability(universe, selected.loc[selected.ev.astype(float).gt(.05) & selected.ev.astype(float).le(.10)], tax_rate=.12)
        selected["selection_rule"] = name
        policy_rows.append(selected)
    if selected_reliability.__globals__["SEED"] != SEED or selected_reliability.__globals__["DRAWS"] != DRAWS:
        raise ValueError("Shared reliability settings differ from frozen protocol")
    scored = {"baseline_full_primary": _scores(primary, "p_baseline"),
              "baseline_common_coverage": _scores(primary.loc[common], "p_baseline"),
              "context_common_coverage": _scores(primary.loc[common], "p"),
              "market_common_coverage": _scores(primary.loc[common], "q_consensus_a"),
              "context_all_available": _scores(new_primary, "p"),
              "market_all_context_available": _scores(new_primary, "q_consensus_a"),
              "context_minus_baseline_common": _paired(primary, common, "p", "p_baseline"),
              "context_minus_market_common": _paired(primary, common, "p", "q_consensus_a")}
    primary_changes = primary[["row_id", "origin_id", "p_baseline", "p", "y", "q_consensus_a", "month", "edition_proxy", "context_primary_eligible"]].merge(
        changes[["origin_id", "changed_roster", "context_same_actual_players", "hindsight_same_players", "canonical_same_cutoff_day"]], on="origin_id", validate="one_to_one")
    controls = {}
    control_masks = {"changed_lineup_retained": primary_changes.changed_roster & primary_changes.context_primary_eligible,
                     "unchanged_lineup_retained": ~primary_changes.changed_roster & primary_changes.context_primary_eligible,
                     "same_actual_players_same_cutoff_retained": primary_changes.context_same_actual_players & primary_changes.canonical_same_cutoff_day & primary_changes.context_primary_eligible}
    for label, mask in control_masks.items():
        controls[label] = {"baseline": _scores(primary_changes.loc[mask], "p_baseline"), "context": _scores(primary_changes.loc[mask], "p"),
                           "market": _scores(primary_changes.loc[mask], "q_consensus_a"),
                           "context_minus_baseline": _paired(primary_changes, mask, "p", "p_baseline"),
                           "context_minus_market": _paired(primary_changes, mask, "p", "q_consensus_a")}
    unchanged = control_masks["unchanged_lineup_retained"]
    unchanged_max = float(abs(primary_changes.loc[unchanged, "p"] - primary_changes.loc[unchanged, "p_baseline"]).max())
    if unchanged_max > EPSILON:
        raise ValueError("Unchanged lineups changed frozen predictions")
    case_ids = {"81029", "80660", "80743", "78657"}
    cases = changes.loc[changes.golgg_match_id.astype(str).isin(case_ids) & changes.cohort.eq("oddspapi_common") & changes.horizon.eq("T-1h")]
    result = {"status": "COMPLETE_CONTEXT_REPAIR_EVALUATION", "run_id": protocol["run_id"],
              "coverage": {"origins": len(context), "baseline_origins_available": int(baseline_origins.sum()),
                           "context_origins_available": int(available_origins.sum()),
                           "lost_origins": int((baseline_origins & ~available_origins).sum()), "restored_origins": int((~baseline_origins & available_origins).sum()),
                           "opportunity_inventory": len(revised), "baseline_primary": len(primary), "common_primary": int(common.sum()),
                           "lost_primary": int((~common).sum()), "context_primary_available": len(new_primary),
                           "restored_primary": int((new_primary.audit_primary_eligible.eq(False)).sum()),
                           "origin_missing_reasons": context.loc[~available_origins, "missing_reason"].value_counts(dropna=False).to_dict(),
                           "unchanged_lineup_max_probability_difference": unchanged_max},
              "proper_scores": scored, "fixed_selections": fixed_stats, "reselected_frozen_policies": policies, "controls": controls,
              "limitations": ["Archived exact competition-label scenario; no certified pre-match announcements in this input.",
                              "Policy returns are unit-stake diagnostics, not an executable ledger or Kelly replay.",
                              "Bookmaker binding remains unverified; no new quote or publication certification.",
                              "Missing context forecasts are abstentions, not wins or zero loss.",
                              "Historical development data; no independent prospective confirmation."]}
    output.mkdir(parents=True)
    revised.to_parquet(output / "opportunities.parquet", index=False)
    changes.to_parquet(output / "roster_changes.parquet", index=False)
    frozen.to_parquet(output / "fixed_selections.parquet", index=False)
    pd.concat(policy_rows, ignore_index=True).to_parquet(output / "policy_selections.parquet", index=False)
    primary_changes.to_parquet(output / "primary_changes.parquet", index=False)
    coverage.to_csv(output / "coverage.csv", index=False)
    cases.to_csv(output / "named_cases.csv", index=False)
    result["output_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in output.iterdir() if path.is_file()}
    save_json(output / "results.json", result)
    return result
