"""Frozen-model input repair evaluation, exposed through the unified betting CLI.

Historical prior lineups and provider outcome conventions remain scenarios.
Filtering unavailable inputs is reported separately from same-origin score gains.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import gc
import gzip
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import ev_asof_replay as replay
from src.analysis.bootstrap import block_ratio_interval
from src.analysis.ev_approach_policies import select_policy_bets, simulate_portfolio
from src.analysis.probability_metrics import binary_log_loss_vector

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["cohort", "fixture_id", "horizon"]
DRAWS = 5000
SEED = 20260930


def _source(config: dict, key: str) -> Path:
    return ROOT / config["sources"][key]["path"]


def _load_inputs(path: Path) -> dict:
    config = json.loads(path.read_text())
    for name, record in config["sources"].items():
        replay._verify_hash(ROOT / record["path"], record["sha256"], name)
    replay._verify_hash(ROOT / config["protocol_path"], config["protocol_sha256"], "repair protocol")
    protocol = json.loads((ROOT / config["protocol_path"]).read_text())
    for name, record in protocol["sources"].items():
        if name != "asof_replay_source":
            replay._verify_hash(ROOT / record["path"], record["sha256"], name)
    return config


def _fresh(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite repair evidence: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def attach_repaired_predictions(opportunities: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Join canonical probabilities by identity, retaining unavailable inventory."""
    if predictions.duplicated(KEYS).any():
        raise ValueError("Repaired forecast keys must be unique")
    fields = [*KEYS, "p_golgg", "status", "missing_reason"]
    clocks = [column for column in ("feature_source_max_day", "cutoff_day") if column in predictions]
    fields.extend(clocks)
    joined = opportunities.merge(
        predictions[fields].rename(columns={"status": "repair_status", "missing_reason": "repair_missing_reason", **{column: f"repair_{column}" for column in clocks}}),
        on=KEYS, how="left", sort=False, validate="many_to_one",
    )
    joined["p_before_repair"] = joined["p"]
    joined["p"] = joined["p_golgg"].where(joined["repair_status"].eq("available"))
    joined["forecast_available"] = joined["repair_status"].eq("available") & joined["p"].between(0, 1, inclusive="neither")
    if len(clocks) == 2:
        source_day = pd.to_datetime(joined["repair_feature_source_max_day"], format="%Y-%m-%d", errors="coerce")
        cutoff_day = pd.to_datetime(joined["repair_cutoff_day"], format="%Y-%m-%d", errors="coerce")
        empty_history = joined["repair_feature_source_max_day"].isna()
        joined["history_available"] = cutoff_day.notna() & (empty_history | source_day.lt(cutoff_day))
    joined["available"] = joined["forecast_available"]
    for field in ("mapping_available", "history_available", "quote_available"):
        joined["available"] &= joined[field].fillna(False).astype(bool)
    missing = joined["repair_missing_reason"].fillna("missing_repaired_prediction")
    joined.loc[~joined["forecast_available"], "exclusion_reason"] = missing
    bad_clock = joined["forecast_available"] & ~joined["history_available"].fillna(False).astype(bool)
    joined.loc[bad_clock, "exclusion_reason"] = "invalid_or_nonprior_feature_clock"
    joined.loc[joined["available"], "exclusion_reason"] = None
    return joined


def _blocks(frame: pd.DataFrame) -> pd.Series:
    dates = frame["date"]
    if "provider_day" in frame:
        dates = dates.combine_first(frame["provider_day"])
    result = pd.to_datetime(dates, utc=True, errors="coerce").dt.strftime("%Y-%m")
    if result.isna().any():
        raise ValueError("Evaluation universe contains an unknown calendar month")
    return result


def selected_reliability(universe: pd.DataFrame, selected: pd.DataFrame, *, tax_rate: float) -> dict:
    """Evaluate selections while preserving zero-bet months in the bootstrap."""
    if universe["row_id"].duplicated().any() or selected["row_id"].duplicated().any():
        raise ValueError("A reliability portfolio requires unique opportunity keys")
    if not set(selected["row_id"]).issubset(set(universe["row_id"])):
        raise ValueError("Selected bets fall outside the declared universe")
    scores = selected[["row_id"]].copy()
    expected = selected["p"].astype(float) * selected["odds"].astype(float) * (1 - tax_rate) - 1
    actual = np.where(selected["won"].astype(bool), selected["odds"].astype(float) * (1 - tax_rate) - 1, -1.)
    scores["ev_error"] = expected.to_numpy() - actual
    scores["probability_error"] = selected["p"].astype(float) - selected["won"].astype(float)
    scores["selected"] = 1.
    aligned = universe[["row_id"]].merge(scores, on="row_id", how="left", validate="one_to_one")
    aligned[["ev_error", "probability_error", "selected"]] = aligned[["ev_error", "probability_error", "selected"]].fillna(0.)
    months = _blocks(universe)
    interval = block_ratio_interval(aligned["ev_error"], aligned["selected"], months, DRAWS, SEED)
    probability_interval = block_ratio_interval(aligned["probability_error"], aligned["selected"], months, DRAWS, SEED)
    reliable = interval["ci_lower"] is not None and interval["ci_lower"] >= -.02 and interval["ci_upper"] <= .02
    return {
        "count": len(selected), "wins": int(selected["won"].sum()),
        "expected_wins": float(selected["p"].astype(float).sum()),
        "mean_predicted_ev": float(expected.mean()) if len(selected) else None,
        "mean_realized_return": float(np.mean(actual)) if len(selected) else None,
        "ev_overstatement_interval": interval, "probability_overstatement_interval": probability_interval,
        "universe_months": int(months.nunique()), "reliable": reliable,
        "assessment": "WITHIN_TOLERANCE" if reliable else ("NOT_ESTIMABLE" if interval["ci_lower"] is None else "NOT_WITHIN_TOLERANCE"),
    }


def replay_inputs(config: dict, research_root: Path | None, roster_evidence: Path | None) -> dict:
    from src.analysis.roster_provenance import load_roster_evidence

    output = ROOT / config["prediction_dir"] / "native_repaired.parquet"
    receipt_path = ROOT / config["report_dir"] / "replay_receipt.json"
    _fresh(output, receipt_path)
    root = replay._resolve_research_root(research_root)
    legacy_protocol = ROOT / "data/05_model_input/ev_all_approaches/ev_all_approaches_20260927_220902/protocol.json"
    protocol = replay._verify_protocol(legacy_protocol)
    bank, releases, bank_receipt, bank_hashes = replay._verify_corrected_bank(root, protocol, ROOT / replay.RAW_HISTORY_RELATIVE)
    original, bank_arrays, bank_temporal = replay._read_bank_validation_data(bank, bank_receipt)
    reference = pd.read_parquet(_source(config, "native_control"))
    reference = reference.loc[reference["scenario"].eq("prior_roster")].copy()
    origins = reference[[*KEYS, "origin_id", "golgg_match_id", "decision_at"]].copy()
    origins["row_id"] = origins["origin_id"]
    origins["scenario_decision_at"] = origins.pop("decision_at")
    origins["decision_at"] = pd.NaT
    origins["information_status"] = "historical_scenario"
    origins["scenario_timestamp_assumption"] = "Frozen reconstructed origin; not an observed decision or historical forecast issuance."
    common = pd.read_parquet(_source(config, "quote_panel"), columns=["fixture_id", "horizon", "start_cutoff_at"])
    common = common.drop_duplicates()
    if common.duplicated(["fixture_id", "horizon"]).any():
        raise ValueError("Conflicting common-cutoff start times")
    origins = origins.merge(common, on=["fixture_id", "horizon"], how="left", validate="many_to_one")
    parents = pd.read_parquet(_source(config, "parent_opportunities"))
    parents = parents.loc[parents["cohort"].eq("original_export_at_frozen_origins"), ["row_id", "start_at"]]
    origins = origins.merge(parents, on="row_id", how="left", validate="many_to_one")
    origins["start_at"] = origins["start_at"].combine_first(origins["start_cutoff_at"])
    origins = origins.drop(columns="start_cutoff_at")
    metadata = original[["golgg_match_id", "team1_id", "team2_id", "best_of", "date"]].copy()
    evidence = load_roster_evidence(roster_evidence)
    result = replay._build_replay_arrays(
        origins, metadata, root, ROOT / replay.RAW_HISTORY_RELATIVE, releases,
        validation_controls=replay._select_oracle_controls(original),
        bank_arrays=bank_arrays, bank_temporal=bank_temporal, roster_evidence=evidence,
    )
    del original, bank_arrays, bank_temporal
    gc.collect()
    frame = result.targets.copy()
    for column in ("p_golgg", "p_no_w20", "p_no_organization"):
        frame[column] = np.nan
    frame["status"] = "unavailable"
    native = replay._activate_native_predictor(root)
    baseline_receipt = json.loads(_source(config, "native_receipt").read_text())
    model_checks = {}
    for year in sorted(frame.loc[frame["feature_available"], "model_year"].dropna().astype(int).unique()):
        if year not in replay.MODEL_YEARS:
            raise ValueError(f"Unsupported frozen model year: {year}")
        positions = np.flatnonzero((frame["model_year"].eq(year) & frame["feature_available"]).to_numpy())
        model = native.CausalA0Predictor(int(year))
        expected = baseline_receipt["controls"]["annual_model_hashes"][str(year)]["artifact_hashes"]
        model_prefix = f"{replay.CAUSAL_ARCHIVE_RELATIVE}/data/06_models/causal_a0/{year}/"
        current_hashes = {str(name).split(model_prefix, 1)[-1]: value for name, value in model.hashes.items()}
        expected_hashes = {str(name).split(model_prefix, 1)[-1]: value for name, value in expected.items()}
        if current_hashes != expected_hashes:
            raise ValueError(f"Model artifact identities changed for {year}")
        for offset in range(0, len(positions), 128):
            indices = positions[offset:offset + 128]
            raw, temporal = replay._batch_features(result.raw, result.hist, indices)
            best_of = frame.iloc[indices]["best_of"].to_numpy(dtype=np.int64)
            values = np.asarray(model.predict(raw, temporal, best_of), dtype=float)
            if values.shape != (len(indices), 3) or not np.isfinite(values).all() or not ((values > 0) & (values < 1)).all():
                raise ValueError("Invalid native full-series probabilities")
            frame.loc[indices, ["p_golgg", "p_no_w20", "p_no_organization"]] = values
            frame.loc[indices, "status"] = "available"
        indices = positions[:3]
        raw, temporal = replay._batch_features(result.raw, result.hist, indices)
        best_of = frame.iloc[indices]["best_of"].to_numpy(dtype=np.int64)
        smoke = replay._native_smoke(model, raw, temporal, best_of)
        model_checks[str(year)] = {"artifact_hashes": dict(model.hashes), "smoke": smoke}
    dependencies = {}
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve().is_relative_to(root):
            path = Path(file).resolve()
            dependencies[str(path.relative_to(root))] = replay._sha256(path)
    aligned = pd.read_parquet(_source(config, "aligned_control"))
    pairs = frame.merge(aligned[[*KEYS, "status", "p_golgg", "p_no_w20", "p_no_organization"]], on=KEYS, validate="one_to_one", suffixes=("", "_control"))
    available = pairs["status"].eq("available") & pairs["status_control"].eq("available")
    differences = {column: float((pairs.loc[available, column] - pairs.loc[available, column + "_control"]).abs().max()) for column in ("p_golgg", "p_no_w20", "p_no_organization")}
    if not evidence["identity_links"] and not evidence["announcements"] and any(value > 2e-6 for value in differences.values()):
        raise ValueError(f"Permanent alignment repair differs from controlled correction: {differences}")
    if frame["decision_at"].notna().any() or not frame["decision_time_source"].eq("scenario").all():
        raise ValueError("A retrospective origin was mislabeled as an actual decision")
    frame["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    frame["replay_contract_version"] = replay.REPLAY_CONTRACT_VERSION
    for column in frame.columns:
        if any(isinstance(value, (dict, list, tuple, np.ndarray)) for value in frame[column]):
            frame[column] = frame[column].map(lambda value: json.dumps(replay._json_safe(value), allow_nan=False))
    frame.to_parquet(output, index=False)
    receipt = {
        "status": "PASS_FROZEN_NATIVE_REPLAY", "run_id": config["run_id"],
        "replay_contract_version": replay.REPLAY_CONTRACT_VERSION,
        "artifact_created_at_utc": frame["generated_at_utc"].iloc[0],
        "decision_time_source": "scenario",
        "replay_source_sha256": replay._sha256(Path(replay.__file__)),
        "bank_hashes": bank_hashes, "feature_source_hashes": result.feature_source_hashes,
        "runtime_archive_dependency_hashes": dependencies, "models": model_checks,
        "replay": result.validation_receipt, "release": result.release_audit,
        "rows": len(frame), "available_rows": int(frame["status"].eq("available").sum()),
        "unavailable_reasons": frame.loc[frame["status"].ne("available"), "missing_reason"].value_counts().to_dict(),
        "aligned_control_common_rows": int(available.sum()), "aligned_control_max_abs_difference": differences,
        "roster_evidence_hashes": evidence["input_hashes"], "training_performed": False,
        "historical_issuance_certified": False, "output_sha256": replay._sha256(output),
    }
    replay._write_json(receipt_path, receipt)
    return receipt


def reconstruct_quotes(config: dict) -> dict:
    from src.analysis.quote_provenance import extract_winner_quote

    output = ROOT / config["report_dir"] / "quote_provenance.parquet"
    receipt_path = output.with_name("quote_receipt.json")
    _fresh(output, receipt_path)
    coverage = pd.read_csv(_source(config, "quote_coverage"), dtype={"fixture_id": str})
    raw_coverage = pd.read_csv(_source(config, "raw_bookmaker_coverage"), dtype={"fixture_id": str}, keep_default_na=False)
    fixture_rows = json.loads(_source(config, "raw_fixtures").read_text())
    fixtures = {str(row["fixtureId"]): row for row in fixture_rows}
    expected_hashes = defaultdict(set)
    with _source(config, "raw_history_receipts").open() as stream:
        for line in stream:
            receipt = json.loads(line)
            expected_hashes[receipt["path"]].add(receipt["sha256"])
    rows = coverage.merge(raw_coverage[["fixture_id", "bookmaker", "path"]], on=["fixture_id", "bookmaker"], validate="many_to_one")
    panel = pd.read_parquet(_source(config, "quote_panel"))
    cutoffs = panel[["fixture_id", "horizon", "evaluation_cutoff_at"]].drop_duplicates()
    rows = rows.merge(cutoffs, on=["fixture_id", "horizon"], how="left", validate="many_to_one")
    out = []
    hashes = {}
    for relative, group in rows.groupby("path", sort=True):
        path = ROOT / relative
        sha = replay._sha256(path)
        if sha not in expected_hashes[relative]:
            raise ValueError(f"Raw quote hash does not match download receipt: {relative}")
        hashes[relative] = sha
        with gzip.open(path, "rt") as stream:
            payload = json.load(stream)
        for row in group.to_dict("records"):
            cutoff = row["evaluation_cutoff_at"]
            fixture = fixtures[row["fixture_id"]]
            if pd.isna(cutoff):
                starts = [pd.Timestamp(fixture[key]) for key in ("startTime", "trueStartTime") if fixture.get(key)]
                if not starts or any(value.tzinfo is None for value in starts):
                    raise ValueError("Fixture start is missing or lacks timezone")
                start = min(starts)
                cutoff = start - pd.Timedelta(hours=1 if row["horizon"] == "T-1h" else 24)
            quote = extract_winner_quote(payload, row["bookmaker"], cutoff_at=cutoff, fixture=fixture)
            record = {"fixture_id": row["fixture_id"], "bookmaker": row["bookmaker"], "horizon": row["horizon"],
                      "evaluation_cutoff_at": cutoff, "source_path": relative, "source_sha256": sha,
                      "available": quote is not None and quote["odds_1"] is not None and quote["odds_2"] is not None,
                      "prior_available": bool(row["available"]), "prior_reason": row["reason"]}
            if quote is not None:
                # The receipt-backed raw file is the lossless archive. Keep only
                # selected-state evidence here, not repeated copies of its future history.
                omitted = {"source_market_evidence", "source_outcome_evidence_1", "source_outcome_evidence_2", "state_1", "state_2"}
                record.update({key: value for key, value in quote.items() if key not in omitted})
                for side in (1, 2):
                    state = quote[f"state_{side}"]
                    record[f"state_status_{side}"] = state["status"]
                    record[f"latest_state_{side}"] = state["latest_states"]
                if quote["source_fixture_id"] is not None and str(quote["source_fixture_id"]) != str(row["fixture_id"]):
                    record["available"] = False
                    record["unavailable_reason"] = "source_fixture_mismatch"
                elif not record["available"]:
                    record["unavailable_reason"] = f"outcome_1:{quote['state_1']['status']};outcome_2:{quote['state_2']['status']}"
            else:
                record["unavailable_reason"] = "bookmaker_or_full_winner_market_missing"
            out.append(record)
    frame = pd.DataFrame(out)
    if frame.duplicated(["fixture_id", "bookmaker", "horizon"]).any():
        raise ValueError("Quote provenance lost unique fixture/book/origin identity")
    compared = panel.merge(frame, on=["fixture_id", "bookmaker", "horizon"], suffixes=("_old", ""), validate="one_to_one")
    for column in ("odds_1", "odds_2"):
        if not np.allclose(compared.loc[compared["available"], column + "_old"], compared.loc[compared["available"], column], atol=1e-12, rtol=0):
            raise ValueError(f"Reconstructed quote prices changed unexpectedly: {column}")
    for column in frame.columns:
        if any(isinstance(value, (dict, list, tuple)) for value in frame[column]):
            frame[column] = frame[column].map(lambda value: json.dumps(replay._json_safe(value), allow_nan=False))
    frame.to_parquet(output, index=False)
    receipt = {"status": "PASS_RAW_STATE_RECONSTRUCTION", "rows": len(frame), "available": int(frame["available"].sum()),
               "raw_files_hashed": len(hashes), "raw_file_hashes": hashes,
               "prior_available_now_unavailable": int((frame["prior_available"] & ~frame["available"]).sum()),
               "participant_binding_status_counts": frame.loc[frame["available"], "participant_binding_status"].value_counts().to_dict(),
               "admission_status_counts": frame["quote_admission_status"].value_counts(dropna=False).to_dict(),
               "same_price_pairs_checked": int(compared["available"].sum()), "output_sha256": replay._sha256(output),
               "execution_certified": False}
    replay._write_json(receipt_path, receipt)
    return receipt


def _quote_contract(frame: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["quote_contract_ok"] = True
    result["bookmaker_binding_verified"] = True
    columns = ["fixture_id", "horizon", "bookmaker", "available", "odds_1", "odds_2", "participant_binding_status", "quote_admission_status"]
    for side in ("a", "b"):
        renamed = quotes[columns].rename(columns={column: f"proof_{side}_{column}" for column in columns if column not in ("fixture_id", "horizon")})
        result = result.merge(renamed, left_on=["fixture_id", "horizon", f"bookmaker_{side}"], right_on=["fixture_id", "horizon", f"proof_{side}_bookmaker"], how="left", validate="many_to_one")
        provider_one = result["provider_team1_is_a"].eq(True) if side == "a" else result["provider_team1_is_a"].eq(False)
        price = result[f"proof_{side}_odds_1"].where(provider_one, result[f"proof_{side}_odds_2"])
        matches = np.isclose(price.astype(float), result[f"odds_{side}"].astype(float), atol=1e-12, rtol=0, equal_nan=False)
        result["quote_contract_ok"] &= result[f"proof_{side}_available"].fillna(False).astype(bool) & matches
        result["bookmaker_binding_verified"] &= result[f"proof_{side}_quote_admission_status"].eq("admitted_with_bookmaker_evidence")
    result.loc[~result["quote_contract_ok"] & result["available"], "exclusion_reason"] = "quote_state_or_binding_unresolved"
    result["available"] &= result["quote_contract_ok"]
    return result


def _select(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "market_consensus_raw_ev_gt_5":
        return select_policy_bets(frame, probability_col="q", policy="P0", tax_rate=.12, min_ev=.05, qualification="raw")
    if policy == "raw_ev_gt_5":
        return select_policy_bets(frame, policy="P0", tax_rate=.12, min_ev=.05, qualification="raw")
    from betting_app.services.bet_qualification_service import is_bet_eligible, qualification_tier

    gated = frame.copy()
    for side in ("a", "b"):
        allow = []
        for row in frame[["p", "q", "odds_a", "odds_b", "league", "date", "best_of"]].itertuples(index=False):
            p = float(row.p) if side == "a" else 1 - float(row.p)
            q = float(row.q) if side == "a" else 1 - float(row.q)
            eligible, _, _ = is_bet_eligible(
                prob_model=p, prob_conservative=None, uncertainty_required=False,
                odds=float(row.odds_a if side == "a" else row.odds_b), prob_market_novig=q,
                tax_rate=.12, min_ev_net=.05, max_ev_net=None,
                competition_tier=qualification_tier(row.league, row.date),
                best_of=int(row.best_of), rating_disagreement=None, prob_market_close_novig=None,
            )
            allow.append(eligible)
        gated[f"allow_{side}"] = allow
    return select_policy_bets(gated, policy="P0", tax_rate=.12, min_ev=0., qualification="raw")


def _cash_oracle(ledger: pd.DataFrame, summary: dict, tax: float) -> dict:
    money = lambda value: Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    cash, total, turnover = money(10000), money(0), money(0)
    events = defaultdict(lambda: {"settle": [], "decision": []})
    rows = ledger.to_dict("records")
    for index, row in enumerate(rows):
        events[pd.Timestamp(row["decision_effective_at"])]["decision"].append(index)
        if row["matured"]:
            events[pd.Timestamp(row["settled_at"])]["settle"].append(index)
    reserved = {}
    peak, max_dd, max_dd_pct = cash, money(0), 0.
    for timestamp in sorted(events):
        for index in events[timestamp]["settle"]:
            row = rows[index]
            stake = reserved.pop(index)
            payout = money(stake * Decimal(str(row["odds"])) * (1 - Decimal(str(tax)))) if row["won"] else money(0)
            profit = payout - stake
            if profit != money(row["pnl"]):
                raise AssertionError("Independent settled payout mismatch")
            cash += payout
            total += profit
        for index in events[timestamp]["decision"]:
            row = rows[index]
            if cash != money(row["free_capital_before"]):
                raise AssertionError("Independent free capital mismatch")
            if row["placed"]:
                stake = money(row["stake"])
                if not 0 < stake <= cash:
                    raise AssertionError("Stake exceeds independently available cash")
                cash -= stake
                turnover += stake
                reserved[index] = stake
            reserve = sum(reserved.values(), money(0))
            if (cash, reserve, cash + reserve) != tuple(money(row[key]) for key in ("free_capital_after", "reserved_stake_after", "equity_after")):
                raise AssertionError("Independent capital or reserve mismatch")
        equity = cash + sum(reserved.values(), money(0))
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        max_dd_pct = max(max_dd_pct, float((peak - equity) / peak) if peak else 0.)
    if reserved or total != money(summary["total_pnl"]) or turnover != money(summary["total_staked"]):
        raise AssertionError("Independent total money mismatch")
    if max_dd != money(summary["max_drawdown"]) or abs(max_dd_pct - summary["max_drawdown_pct"]) > 1e-12:
        raise AssertionError("Independent drawdown mismatch")
    return {"status": "PASS" if len(rows) else "PASS_EMPTY", "rows": len(rows), "events": len(events)}


def _portfolio(universe: pd.DataFrame, selected: pd.DataFrame, context: dict, strategy: str) -> tuple[dict, pd.DataFrame]:
    chosen = selected.copy()
    chosen["settle_at"] = pd.NaT
    ledger, summary = simulate_portfolio(chosen, strategy=strategy, tax_rate=.12, flat_stake=100., initial_bankroll=10000., kelly_fraction=.25, max_fraction=.025)
    oracle = _cash_oracle(ledger, summary, .12)
    realized = ledger.loc[ledger["placed"] & ledger["matured"]]
    joined = universe[["row_id"]].merge(realized[["row_id", "stake", "pnl"]], on="row_id", how="left", validate="one_to_one")
    joined[["stake", "pnl"]] = joined[["stake", "pnl"]].fillna(0.)
    interval = block_ratio_interval(joined["pnl"].astype(float), joined["stake"].astype(float), _blocks(universe), DRAWS, SEED)
    tax0_profit = sum((Decimal(str(row.stake)) * Decimal(str(row.odds))).quantize(Decimal('.01'), rounding=ROUND_HALF_UP) - Decimal(str(row.stake)) if row.won else -Decimal(str(row.stake)) for row in realized.itertuples())
    summary.update(context)
    summary.update({"strategy": strategy, "independent_cash_oracle": oracle, "roi_interval": interval,
                    "reliability": selected_reliability(universe, selected, tax_rate=.12),
                    "near_5pct": selected_reliability(universe, selected.loc[selected["ev"].astype(float).gt(.05) & selected["ev"].astype(float).le(.10)], tax_rate=.12),
                    "same_stakes_tax0_pnl": float(tax0_profit), "same_stakes_tax0_roi": float(tax0_profit) / summary["total_staked"] if summary["total_staked"] else None,
                    "certified_close_or_clv": False})
    for field, value in {**context, "strategy": strategy}.items():
        ledger[field] = value
    return summary, ledger


def evaluate_repaired(config: dict) -> dict:
    output = ROOT / config["report_dir"]
    _fresh(*(output / name for name in ("results.json", "summary.csv", "ledger.parquet", "opportunities.parquet", "parent_opportunities.parquet", "proper_scores.csv", "coverage.csv")))
    prediction_path = ROOT / config["prediction_dir"] / "native_repaired.parquet"
    quote_path = output / "quote_provenance.parquet"
    for path, receipt_name in ((prediction_path, "replay_receipt.json"), (quote_path, "quote_receipt.json")):
        receipt = json.loads((output / receipt_name).read_text())
        replay._verify_hash(path, receipt["output_sha256"], receipt_name)
    predictions = pd.read_parquet(prediction_path)
    quotes = pd.read_parquet(quote_path)
    baseline = pd.read_parquet(_source(config, "native_opportunities"))
    baseline = baseline.loc[baseline["scenario"].eq("prior_roster")].copy()
    repaired = attach_repaired_predictions(baseline, predictions)
    repaired = _quote_contract(repaired, quotes)
    conflicts = pd.read_parquet(_source(config, "quote_conflicts"))
    repaired = repaired.merge(conflicts[["fixture_id", "horizon", "conflict_q25"]], on=["fixture_id", "horizon"], how="left", validate="many_to_one")
    repaired["conflict_q25"] = repaired["conflict_q25"].fillna(False).astype(bool)
    repaired["execution_evidenced"] = False  # Newly generated forecasts were not issued at historical decision times.
    reports, ledgers, score_rows = [], [], []
    policies = ("raw_ev_gt_5", "current_application_filters", "market_consensus_raw_ev_gt_5")
    for (horizon, source), inventory in repaired.groupby(["horizon", "source"], sort=True):
        base = inventory.loc[inventory["available"]].copy()
        for quality in ("as_supplied_diagnostic", "exclude_q25_conflicts", "execution_evidenced"):
            universe = base if quality == "as_supplied_diagnostic" else base.loc[~base["conflict_q25"]] if quality == "exclude_q25_conflicts" else base.loc[base["execution_evidenced"]]
            for arm in ("repaired", "original_same_origins"):
                frame = universe.copy()
                if arm == "original_same_origins":
                    frame = frame.loc[frame["p_before_repair"].between(0, 1, inclusive="neither")].copy()
                    frame["p"] = frame["p_before_repair"]
                if len(frame):
                    y, p = frame["y"].to_numpy(float), frame["p"].to_numpy(float)
                    market = frame["q_consensus_a"] if source == "best_price" else frame["q_book_a"]
                    score = {"horizon": horizon, "source": source, "quality": quality, "arm": arm, "n": len(frame),
                             "log_loss": float(binary_log_loss_vector(y, p).mean()), "brier": float(np.mean((p-y)**2)),
                             "market_log_loss": float(binary_log_loss_vector(y, market.to_numpy(float)).mean()),
                             "delta_vs_market": block_ratio_interval(binary_log_loss_vector(y,p)-binary_log_loss_vector(y,market.to_numpy(float)),np.ones(len(frame)),_blocks(frame),DRAWS,SEED)}
                    if arm == "repaired":
                        paired = frame.loc[frame["p_before_repair"].between(0, 1, inclusive="neither")]
                        paired_y = paired["y"].to_numpy(float)
                        before_loss = binary_log_loss_vector(paired_y, paired["p_before_repair"].to_numpy(float))
                        after_loss = binary_log_loss_vector(paired_y, paired["p"].to_numpy(float))
                        score["same_origin_repair_effect"] = {
                            "n": len(paired),
                            "original_log_loss": float(before_loss.mean()) if len(paired) else None,
                            "repaired_log_loss": float(after_loss.mean()) if len(paired) else None,
                            "delta_log_loss": block_ratio_interval(after_loss-before_loss, np.ones(len(paired)), _blocks(paired), DRAWS, SEED),
                        }
                    score_rows.append(score)
                for policy in policies:
                    if arm == "original_same_origins" and policy == "market_consensus_raw_ev_gt_5":
                        continue
                    selected = _select(frame, policy)
                    changed = frame.copy()
                    changed["y"] = 1 - changed["y"]
                    selected_changed = _select(changed, policy)
                    columns = ["row_id", "side", "p", "odds"]
                    pd.testing.assert_frame_equal(selected[columns].reset_index(drop=True), selected_changed[columns].reset_index(drop=True))
                    for strategy in ("flat", "quarter_kelly"):
                        context = {"cohort": "oddspapi_common", "horizon": horizon, "source": source, "quality": quality, "arm": arm, "policy": policy, "inventory_rows": len(inventory), "available_rows": len(frame)}
                        summary, ledger = _portfolio(inventory, selected, context, strategy)
                        reports.append(summary)
                        ledgers.append(ledger)
    parents = pd.read_parquet(_source(config, "parent_opportunities"))
    parent_predictions = predictions.loc[predictions["cohort"].eq("parent_origins"), ["origin_id", "p_golgg", "status", "missing_reason"]]
    parents = parents.merge(parent_predictions, left_on="row_id", right_on="origin_id", how="left", validate="many_to_one", suffixes=("", "_repair"))
    for (cohort, horizon), inventory in parents.groupby(["cohort", "horizon"], sort=True):
        universe = inventory.loc[inventory["available"] & inventory["status"].eq("available")].copy()
        for arm in ("repaired", "original_same_origins"):
            frame = universe.copy()
            frame["p"] = frame["p_golgg"] if arm == "repaired" else frame["prior_roster"]
            frame = frame.loc[frame["p"].between(0, 1, inclusive="neither")].copy()
            for policy in policies:
                if arm == "original_same_origins" and policy == "market_consensus_raw_ev_gt_5":
                    continue
                selected = _select(frame, policy)
                for strategy in ("flat", "quarter_kelly"):
                    context = {"cohort": cohort, "horizon": horizon, "source": "best_price", "quality": "archived_production_diagnostic", "arm": arm, "policy": policy, "inventory_rows": len(inventory), "available_rows": len(frame)}
                    summary, ledger = _portfolio(inventory, selected, context, strategy)
                    reports.append(summary)
                    ledgers.append(ledger)
    repaired.to_parquet(output / "opportunities.parquet", index=False)
    parents.to_parquet(output / "parent_opportunities.parquet", index=False)
    pd.concat(ledgers, ignore_index=True, sort=False).to_parquet(output / "ledger.parquet", index=False)
    pd.json_normalize(reports).to_csv(output / "summary.csv", index=False)
    pd.json_normalize(score_rows).to_csv(output / "proper_scores.csv", index=False)
    repaired.groupby(["horizon", "source", "available", "exclusion_reason"], dropna=False).size().rename("rows").reset_index().to_csv(output / "coverage.csv", index=False)
    result = {"status": "COMPLETE_DEVELOPMENT_REPAIR_EVALUATION", "run_id": config["run_id"], "portfolios": reports,
              "proper_scores": score_rows, "fixture_inventory": int(repaired["fixture_id"].nunique()),
              "opportunities": len(repaired), "financial_cases": len(reports),
              "monthly_blocking": "Canonical match date, provider_day for unmapped fixtures; all inventory months including no-bet months retained.",
              "interpretation": "Exclusions change coverage, not model accuracy. Original comparisons use identical repaired-available origins. Unverified bookmaker bindings and prior lineups remain diagnostic; no historical issuance certification.",
              "G0": "BLOCKED", "new_fitting": False, "deployment": False}
    replay._write_json(output / "results.json", result)
    return {key: value for key, value in result.items() if key not in ("portfolios", "proper_scores")}


def input_repair_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--stage", choices=("replay", "quotes", "evaluate"), required=True)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--roster-evidence", type=Path)
    args = parser.parse_args(argv)
    config = _load_inputs(args.inputs)
    if args.stage == "replay":
        receipt = replay_inputs(config, args.research_root, args.roster_evidence)
    elif args.stage == "quotes":
        receipt = reconstruct_quotes(config)
    else:
        receipt = evaluate_repaired(config)
    print(json.dumps(replay._json_safe({key: value for key, value in receipt.items() if key not in ("raw_file_hashes", "runtime_archive_dependency_hashes", "release", "replay", "bank_hashes", "feature_source_hashes")}), indent=2, allow_nan=False))
