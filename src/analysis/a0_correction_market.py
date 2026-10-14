"""Nonexecuting fixed/own-selection EV diagnostics for frozen sports forecasts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.bootstrap import block_ratio_interval
from src.analysis.ev_approach_data import _key_series
from src.analysis.model_benchmark import compute_benchmark_summary
from src.analysis.production_market_benchmark import paired_cluster_interval
from src.analysis.research_benchmark import losses, safe
from src.models.a0_correction_experiment import _json, _verified, sha256, write_json


def evaluate_market(protocol_path: Path, predictions_path: Path, output_dir: Path) -> dict:
    protocol = _json(protocol_path)
    _verified(protocol["source"], protocol["source_sha256"])
    quotes = pd.read_parquet(protocol["source"])
    predictions = pd.read_parquet(predictions_path)
    columns = protocol["model_columns"]
    candidate = protocol["candidate_column"]
    comparisons = protocol["comparison_columns"]
    if candidate not in columns or not comparisons or any(
        name not in [*columns, "p_market"] or name == candidate for name in comparisons
    ):
        raise ValueError("Market contrasts require an explicit candidate and distinct declared comparators")
    required = ["golgg_match_id", "team1_id", "team2_id", "date", "best_of", "y",
                "feature_source_max_day", "head_origin", *columns]
    forecasts = predictions[required].copy()
    forecasts["golgg_match_id"] = _key_series(forecasts.golgg_match_id)
    if forecasts.golgg_match_id.isna().any() or forecasts.golgg_match_id.duplicated().any():
        raise ValueError("Forecast identities are incomplete or duplicated")
    for column in columns:
        p = forecasts[column].to_numpy(float)
        if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
            raise ValueError(f"Invalid forecast probabilities: {column}")
    forecasts = forecasts.rename(columns={key: "f_" + key for key in required if key != "golgg_match_id"})
    quotes["golgg_match_id"] = _key_series(quotes.golgg_match_id)
    ledger = quotes.merge(forecasts, on="golgg_match_id", how="left", validate="many_to_one", indicator=True)
    matched = ledger._merge.eq("both")
    for key in ("team1_id", "team2_id"):
        if not _key_series(ledger.loc[matched, key]).eq(_key_series(ledger.loc[matched, "f_" + key])).all():
            raise ValueError(f"Quote/forecast canonical {key} mismatch")
    if not ledger.loc[matched, "date"].astype(str).eq(ledger.loc[matched, "f_date"].astype(str)).all():
        raise ValueError("Quote/forecast canonical date mismatch")
    for key, domain in (("best_of", [1, 3, 5]), ("y", [0, 1])):
        left = pd.to_numeric(ledger.loc[matched, key], errors="coerce")
        right = pd.to_numeric(ledger.loc[matched, "f_" + key], errors="coerce")
        if not left.isin(domain).all() or not left.eq(right).all():
            raise ValueError(f"Quote/forecast canonical {key} mismatch")
    if not np.allclose(ledger.loc[matched, "p_a0"], ledger.loc[matched, "f_p_a0"], atol=1e-12, rtol=0.):
        raise ValueError("Quote panel A0 probabilities are not in canonical orientation")
    if not ledger.provider_team1_is_a.isin([True, False]).all():
        raise ValueError("Quote side orientation is unknown")
    same = ledger.provider_team1_is_a.to_numpy(bool)
    ledger["odds_a"] = np.where(same, ledger.odds_1, ledger.odds_2)
    ledger["odds_b"] = np.where(same, ledger.odds_2, ledger.odds_1)
    odds = ledger[["odds_a", "odds_b"]].to_numpy(float)
    if not np.isfinite(odds).all() or np.any(odds <= 1):
        raise ValueError("Invalid source decimal odds")
    ledger["p_market"] = (1 / ledger.odds_a) / (1 / ledger.odds_a + 1 / ledger.odds_b)
    if not np.allclose(ledger.p_market, ledger.p_book, atol=1e-12, rtol=0.):
        raise ValueError("Reoriented odds disagree with archived canonical market probability")
    cutoff = pd.to_datetime(ledger.evaluation_cutoff_at, utc=True, errors="raise")
    start = pd.to_datetime(ledger.start_cutoff_at, utc=True, errors="raise")
    clocks = [pd.to_datetime(ledger[name], utc=True, errors="raise") for name in
              ("quote_at", "outcome_1_quote_at", "outcome_2_quote_at")]
    validity = pd.to_datetime(ledger.valid_until_exclusive_at, utc=True, errors="coerce")
    source_day = pd.to_datetime(ledger.f_feature_source_max_day, utc=True, errors="coerce")
    fit_origin = pd.to_datetime(ledger.f_head_origin, utc=True, errors="coerce")
    conditions = {
        "different_bookmaker": ledger.bookmaker.eq(protocol["bookmaker"]),
        "different_horizon": ledger.horizon.eq(protocol["horizon"]),
        "missing_forecast": matched,
        "not_strict_prior_day": ledger.strict_prior_day.eq(True),
        "stale_side": ledger.max_side_age_hours.le(protocol["maximum_side_age_hours"]),
        "future_source_feature": source_day.lt(cutoff.dt.floor("D")),
        "head_not_yet_available": fit_origin.le(cutoff),
        "decision_not_prestart": cutoff.lt(start),
        "future_or_unknown_quote_clock": clocks[0].le(cutoff) & clocks[1].le(cutoff) & clocks[2].le(cutoff),
        "inactive_quote_state": validity.isna() | cutoff.lt(validity),
    }
    reason_rows = [[] for _ in range(len(ledger))]
    for name, condition in conditions.items():
        for index in np.flatnonzero(~condition.fillna(False).to_numpy(bool)):
            reason_rows[index].append(name)
    ledger["exclusion_reasons"] = ["|".join(items) for items in reason_rows]
    ledger["admitted_scenario"] = ledger.exclusion_reasons.eq("")
    common = ledger.loc[ledger.admitted_scenario].copy().reset_index(drop=True)
    if common.empty or common.fixture_id.duplicated().any() or common.golgg_match_id.duplicated().any():
        raise ValueError("Common quote scenario is empty or has repeated fixture identities")
    for name in columns:
        common[name] = common["f_" + name]
    monthly = common.date.astype(str).str[:7].to_numpy()
    source_blocks = (common.tournament_name.fillna("unknown").astype(str) + "|" + common.date.astype(str).str[:4]).to_numpy()
    y = common.y.to_numpy(int)
    metrics = {name: asdict(compute_benchmark_summary(y, common[name], probability_epsilon=None))
               for name in [*columns, "p_market"]}
    candidate_loss = losses(y, common[candidate])
    contrasts = {name: paired_cluster_interval(candidate_loss - losses(y, common[name]), monthly,
                 draws=protocol["resamples"], seed=protocol["seed"]) for name in comparisons}
    odds = common[["odds_a", "odds_b"]].to_numpy(float)
    multiplier = odds * (1 - protocol["tax"])
    outcome = np.column_stack([y, 1 - y])
    probabilities = {name: np.column_stack([common[name], 1 - common[name]]) for name in columns}
    ev = {name: p * multiplier - 1 for name, p in probabilities.items()}
    decisions = {name: np.argmax(value, axis=1) for name, value in ev.items()}
    selections = {name: (np.max(value, axis=1) > .05) & ~np.isclose(value[:, 0], value[:, 1], atol=1e-12, rtol=0.)
                  for name, value in ev.items()}
    rows = np.arange(len(common))
    reports = []
    selected_rows = []
    for policy, selectors in (("fixed_a0", ["p_a0"]), ("own", columns)):
        for selector in selectors:
            side = decisions[selector]
            selector_ev = ev[selector][rows, side]
            for band in protocol["bands"]:
                selected = selections[selector].copy()
                if band == "EV_5_to_10pct":
                    selected &= selector_ev <= .1
                denominator = selected.astype(float)
                realized = outcome[rows, side] * multiplier[rows, side] - 1
                for model in (columns if policy == "fixed_a0" else [selector]):
                    expected = ev[model][rows, side]
                    error = expected - realized
                    interval = block_ratio_interval(error * denominator, denominator, monthly,
                                                    draws=protocol["resamples"], seed=protocol["seed"])
                    sensitivity = block_ratio_interval(error * denominator, denominator, source_blocks,
                                                       draws=protocol["resamples"], seed=protocol["seed"])
                    n = int(selected.sum())
                    reliable = interval["ci_lower"] is not None and interval["ci_upper"] is not None and (
                        interval["ci_lower"] >= -protocol["tolerance"] and interval["ci_upper"] <= protocol["tolerance"])
                    reports.append({"policy": policy, "selector": selector, "model": model, "band": band,
                                    "n": n, "wins": int(outcome[rows, side][selected].sum()),
                                    "mean_predicted_net_return": float(expected[selected].mean()) if n else None,
                                    "mean_realized_unit_net_return": float(realized[selected].mean()) if n else None,
                                    "overstatement": interval, "source_label_year_sensitivity": sensitivity,
                                    "within_two_percentage_points": bool(reliable)})
                    for i in np.flatnonzero(selected):
                        selected_rows.append({"fixture_id": str(common.fixture_id.iloc[i]),
                            "golgg_match_id": str(common.golgg_match_id.iloc[i]), "date": str(common.date.iloc[i]),
                            "policy": policy, "selector": selector, "model": model, "band": band,
                            "side": "a" if side[i] == 0 else "b", "odds": float(odds[i, side[i]]),
                            "p": float(probabilities[model][i, side[i]]), "y": int(outcome[i, side[i]]),
                            "expected": float(expected[i]), "realized": float(realized[i]), "error": float(error[i])})
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger.drop(columns="_merge").to_parquet(output_dir / "opportunity_ledger.parquet", index=False)
    common.drop(columns="_merge").to_parquet(output_dir / "common.parquet", index=False)
    pd.DataFrame(selected_rows).to_parquet(output_dir / "selected.parquet", index=False)
    result = {"status": "HISTORICAL_DIAGNOSTIC_ONLY", "source_rows": len(ledger), "common_rows": len(common),
              "excluded_by_reason": {name: int((~value.fillna(False)).sum()) for name, value in conditions.items()},
              "months": len(np.unique(monthly)), "metrics": metrics, "paired_log_loss": contrasts,
              "candidate_column": candidate, "comparison_columns": comparisons,
              "selected_diagnostics": reports, "protocol_sha256": sha256(protocol_path),
              "predictions_sha256": sha256(predictions_path), "source_sha256": protocol["source_sha256"],
              "code_sha256": sha256(__file__), "limitations": protocol["limits"]}
    write_json(output_dir / "results.json", safe(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_market(args.protocol, args.predictions, args.output_dir)
    print(json.dumps({"status": result["status"], "common_rows": result["common_rows"],
                      "months": result["months"], "selected_diagnostics": result["selected_diagnostics"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
