"""Bounded native-C0 betting development; never admits authentic forward data.

Uses existing immutable C0 outputs and provider scenario prices. The CLI extends
our unified evaluator; it does not refit C0, deploy a head or repair source clocks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.analysis.ev_approach_runner import _json_safe, _resolve_research_root
from src.analysis.c0_betting_development_evaluation import evaluate_development
from src.models.ev_approach_models import (
    C0_RECIPES, C0_EPSILON, C0_NATIVE_VERSION, fit_c0_market_head,
    select_c0_market_head, C0MarketHead,
)
from src.models.c0_betting_price_head import fit_price_head, C0PriceHead

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "c0_historical_betting_development_v1"
RECIPES = (*C0_RECIPES, "M-price", "H-price")
LAMBDAS = (.01, .1, 1.)
PHASES = {"TRAIN": ("2026-01-01", "2026-05-01"),
          "CAL": ("2026-05-01", "2026-07-01"),
          "ASSESS": ("2026-07-01", "2026-09-09")}
FOCUS = "a0_data_focus_20260929_192443"
FORWARD = "data/05_model_input/c0_market_study/c0_market_contracts_20261006_01/protocol.json"
CODE = ("src/analysis/c0_betting_development.py",
        "src/analysis/c0_betting_development_evaluation.py",
        "src/models/c0_betting_price_head.py", "src/analysis/ev_approach_runner.py")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(_json_safe(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)+"\n")


def _identity_equal(row, other):
    for column in ("golgg_match_id", "team1_id", "team2_id"):
        if pd.isna(row[column]) or pd.isna(other[column]) or str(row[column]) != str(other[column]):
            raise ValueError(f"Conflicting historical identity: {column}")
    for column, allowed in (("best_of", {1, 3, 5}), ("y", {0, 1})):
        values = (row[column], other[column])
        if (any(isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value not in allowed for value in values)
                or values[0] != values[1]):
            raise ValueError(f"Conflicting historical identity: {column}")


def _utc(value, name):
    if pd.isna(value):
        raise ValueError(f"Missing {name}")
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"Timezone missing for {name}")
    return stamp.tz_convert("UTC")


def build_panel(control, market, quotes, *, max_age_seconds=86400):
    """Retain the full provider denominator; invalid book sets do not fallback.

    A source quote clock is a scenario ordering clock, never a local receipt.
    `native_admitted` means native scenario support here, not G0 admission.
    """
    control = control.copy()
    market = market.copy()
    quotes = quotes.copy()
    for frame in (control, market, quotes):
        frame["golgg_match_id"] = frame["golgg_match_id"].astype(str)
    if control.golgg_match_id.duplicated().any() or market.golgg_match_id.duplicated().any():
        raise ValueError("Duplicate native/market identity")
    if market.fixture_id.duplicated().any():
        raise ValueError("Duplicate provider fixture identity")
    native = control.set_index("golgg_match_id", drop=False)
    t1 = quotes.loc[(quotes.horizon == "T-1h") & quotes.strict_prior_day
        & quotes.golgg_match_id.isin(market.golgg_match_id)]
    if t1.duplicated(["fixture_id", "bookmaker"]).any():
        raise ValueError("Ambiguous latest bookmaker state; no older-state fallback")
    groups = {key: frame for key, frame in t1.groupby("fixture_id", sort=False)}
    records = []
    for _, joined in market.iterrows():
        if joined.golgg_match_id not in native.index:
            raise ValueError("Missing native identity in market denominator")
        source = native.loc[joined.golgg_match_id]
        _identity_equal(joined, source)
        p = source.p_full_full
        if (isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, float, np.number))
                or not np.isfinite(p) or not 0 <= p <= 1
                or joined.control__full__full != p):
            raise ValueError("Wrong native C0 probability identity")
        group = groups.get(joined.fixture_id)
        if group is None or group.empty:
            raise ValueError("Missing bound nominal cutoff; never invent a timestamp")
        cutoff_values = {_utc(value, "evaluation_cutoff_at") for value in group.evaluation_cutoff_at}
        if len(cutoff_values) != 1:
            raise ValueError("Conflicting nominal decision identity")
        cutoff = next(iter(cutoff_values))
        feature_day = pd.Timestamp(source.feature_source_max_day)
        if pd.isna(feature_day) or feature_day.date() >= cutoff.date():
            raise ValueError("Noncausal native feature cutoff")
        phase = next((name for name, (start, end) in PHASES.items()
            if pd.Timestamp(start, tz="UTC") <= cutoff < pd.Timestamp(end, tz="UTC")), None)
        if phase is None:
            raise ValueError("Nominal cutoff outside frozen native 2026 phases")
        release_value = source.get("effective_label_release_day", source.result_day)
        release = pd.Timestamp(release_value)
        if pd.isna(release):
            raise ValueError("Missing daily label release scenario")
        label_bound = PHASES[phase][1] if phase != "ASSESS" else "2026-10-06"
        fit_eligible = release.date() < pd.Timestamp(label_bound).date()
        admitted = []
        invalid_books = []
        for _, quote in group.iterrows():
            _identity_equal(joined, quote)
            if not isinstance(quote.provider_team1_is_a, (bool, np.bool_)):
                raise ValueError("Missing provider side identity")
            start = _utc(quote.start_cutoff_at, "start_cutoff_at")
            times = [_utc(quote[column], column) for column in
                ("quote_at", "outcome_1_quote_at", "outcome_2_quote_at")]
            ages = [(cutoff-time).total_seconds() for time in times]
            valid_until = quote.get("valid_until_exclusive_at")
            expiry = None if pd.isna(valid_until) else _utc(valid_until, "valid_until_exclusive_at")
            odds = [quote.odds_1, quote.odds_2]
            valid_prices = all(not isinstance(value, (bool, np.bool_))
                and isinstance(value, (int, float, np.number)) and np.isfinite(value)
                and value > 1 for value in odds)
            if (cutoff >= start or not valid_prices or any(age < 0 or age > max_age_seconds for age in ages)
                    or (expiry is not None and cutoff >= expiry)):
                invalid_books.append(str(quote.bookmaker))
                continue
            oa, ob = odds if quote.provider_team1_is_a else odds[::-1]
            lineage = {key: quote[key] for key in ("source_path", "source_sha256", "fixture_id",
                "bookmaker", "outcome_1_source_index", "outcome_2_source_index",
                "outcome_1_quote_at", "outcome_2_quote_at")}
            admitted.append({"bookmaker": str(quote.bookmaker), "odds_a": float(oa),
                "odds_b": float(ob), "quote_id": "source-composite:"+_canonical_sha(lineage),
                "identity_kind": "bound_source_composite_not_database_quote_id", "lineage": _json_safe(lineage),
                "source_quote_at": times[0].isoformat(), "scenario_age_seconds": max(ages)})
        admitted.sort(key=lambda quote: (quote["bookmaker"], quote["quote_id"]))
        eligible = len(admitted) >= 2
        q = float(np.mean([quote["odds_b"]/(quote["odds_a"]+quote["odds_b"])
            for quote in admitted])) if eligible else np.nan
        best = {side: sorted(admitted, key=lambda quote: (-quote["odds_"+side], quote["bookmaker"],
            quote["quote_id"]))[0] if eligible else None for side in ("a", "b")}
        record = source.to_dict()
        record.update(row_id=f"{joined.fixture_id}:{joined.golgg_match_id}",
            fixture_id=str(joined.fixture_id), scenario_decision_at=cutoff.isoformat(),
            label_release_day=release.date().isoformat(), phase=phase, fit_eligible=bool(fit_eligible),
            month=cutoff.strftime("%Y-%m"), p_C0=float(p), q=q,
            native_admitted=True, market_eligible=eligible,
            financial_eligible=bool(release.date() < pd.Timestamp("2026-10-06").date()), primary_decision=True,
            settlement_status="retrospective_daily_mature_label_scenario"
                if release.date() < pd.Timestamp("2026-10-06").date() else "unreleased_daily_label",
            quotes=admitted if eligible else [], admitted_book_count=len(admitted),
            invalid_books=invalid_books, market_absence=None if eligible else "fewer_than_two_complete_fresh_scenario_books",
            odds_a=best["a"]["odds_a"] if eligible else np.nan,
            odds_b=best["b"]["odds_b"] if eligible else np.nan,
            bookmaker_a=best["a"]["bookmaker"] if eligible else None,
            bookmaker_b=best["b"]["bookmaker"] if eligible else None,
            quote_id_a=best["a"]["quote_id"] if eligible else None,
            quote_id_b=best["b"]["quote_id"] if eligible else None,
            clock_classification="retrospective_provider_source_state_and_daily_release_scenario_not_issuance")
        if phase == "ASSESS" and not fit_eligible:
            record["y"] = None
            record["financial_eligible"] = False
        records.append(record)
    panel = pd.DataFrame(records).sort_values(["scenario_decision_at", "row_id"]).reset_index(drop=True)
    panel.attrs["monthly_universe"] = sorted(panel.month.unique())
    coverage = {"native_count": len(control), "provider_denominator": len(panel),
        "native_outside_provider_count": int((~control.golgg_match_id.isin(market.golgg_match_id)).sum()),
        "max_age_seconds": max_age_seconds, "phases": {phase: {
            "denominator": int((panel.phase == phase).sum()),
            "common_count": int(((panel.phase == phase) & panel.market_eligible & panel.fit_eligible).sum()),
            "market_absence_count": int(((panel.phase == phase) & ~panel.market_eligible).sum()),
            "late_release_count": int(((panel.phase == phase) & ~panel.fit_eligible).sum())} for phase in PHASES}}
    return panel, coverage


def _source_path(reference, research_root):
    root = PROJECT_ROOT if reference["root"] == "repository" else research_root
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source escapes its declared root")
    return path


def freeze(run_id, research_root=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("Invalid run ID")
    root = _resolve_research_root(research_root)
    stage05 = PROJECT_ROOT / "data/05_model_input/c0_betting_development" / run_id
    evidence = PROJECT_ROOT / "data/08_reporting/c0_betting_development" / run_id
    adequacy = json.loads((evidence / "data_adequacy.json").read_text())
    references = {
        "control": ("repository", f"data/07_model_output/a0_data_focus/{FOCUS}/control.parquet"),
        "market": ("repository", f"data/08_reporting/a0_data_focus/{FOCUS}/market_T1_joined.parquet"),
        "quotes": ("repository", "data/08_reporting/benchmark/a0_bookmakers_20260929_092422/common_cutoff_panel.parquet"),
        "availability": ("research", "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank/availability.jsonl"),
        "release_map": ("research", "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank/release_days.json"),
        "native_config": ("repository", "conf/base/c0_serving.json"),
        "native_protocol": ("repository", f"data/05_model_input/a0_data_focus/{FOCUS}/protocol.json"),
        "data_adequacy": ("repository", str((evidence / "data_adequacy.json").relative_to(PROJECT_ROOT))),
        "nominal_adequacy": ("repository", str((evidence / "data_adequacy_nominal_supplement.json").relative_to(PROJECT_ROOT))),
        "behavior_audit": ("repository", str((evidence / "behavior/behavior_audit.json").relative_to(PROJECT_ROOT))),
        "frozen_forward_protocol": ("repository", FORWARD),
    }
    for index, path in enumerate(adequacy["native_control"]["identity"]["artifact_hashes"]):
        references[f"native_artifact_{index}"] = ("repository", path)
    sources = {}
    for name, (source_root, path) in references.items():
        reference = {"root": source_root, "path": path}
        reference["sha256"] = _sha(_source_path(reference, root))
        sources[name] = reference
    for path, expected in adequacy["native_control"]["identity"]["artifact_hashes"].items():
        if _sha(PROJECT_ROOT/path) != expected:
            raise ValueError("Native artifact differs from audited C0 identity")
    forward = json.loads((PROJECT_ROOT/FORWARD).read_text())
    for path, expected in forward["code_hashes"].items():
        if _sha(PROJECT_ROOT/path) != expected:
            raise ValueError("Existing forward code binding changed")
    protocol = {"schema": SCHEMA, "run_id": run_id, "scope": "EXPOSED_HISTORICAL_DEVELOPMENT_ONLY",
        "frozen_at_actual_utc": datetime.now(timezone.utc).isoformat(),
        "native_identity": f"Causal-C0:{C0_NATIVE_VERSION}", "mode": "full", "epsilon": C0_EPSILON,
        "calendar": {name: list(dates) for name, dates in PHASES.items()},
        "membership_clock": "immutable_evaluation_cutoff_at_UTC_not_target_date",
        "lambdas": list(LAMBDAS), "recipes": list(RECIPES), "formal_fit_budget": 18, "base_fit_budget": 0,
        "max_quote_age_seconds": 86400, "freshness_diagnostic_seconds": 300,
        "minimum_books": 2, "tax": .12, "minimum_net_ev_exclusive": .05,
        "objective": "mean_unclipped_logit_nll_plus_lambda_over_2_centered_squared_l2",
        "price_features": {"M-price": ["inner(logit(q))", "outer(logit(q))", "log(best_odds_B/best_odds_A)"],
            "H-price": ["inner(logit(q))", "outer(logit(q))", "log(best_odds_B/best_odds_A)", "logit(p_C0)"]},
        "centers": {"M-price": [1.,1.,0.], "H-price": [1.,1.,0.,0.]},
        "selection": "unpenalized_clipped_CAL_LL_largest_lambda_within_1e-12_no_refit",
        "primary_candidate": "H-price", "primary_controls": ["M-price", "M-shape"],
        "statistics": {"bootstrap_draws": 5000, "seed": 20261006, "block": "calendar_month",
            "primary_interval_level": .975, "secondary_interval_level": .95,
            "empty_conditional_draw": "entire_interval_NOT_ESTIMABLE_no_finite_subset"},
        "success": "Both primary LL upper<0, Brier nonhigher, coverage nonlower; selected and near-band error CIs wholly±.02; positive cash/opportunity and cash increment versus M-price lower limits. Even PASS is development-only, never operational adoption.",
        "sources": sources, "code_hashes": {path: _sha(PROJECT_ROOT/path) for path in CODE},
        "preserved_forward_code_hashes": forward["code_hashes"],
        "expected_population": {"native": 3103, "provider": 1668, "primary_common": 1619},
        "limitations": ["All 2026 outcomes previously exposed; no protected confirmation.",
            "Real fit time October6 is distinct from hypothetical May1/July1 origins.",
            "Actual first-map rosters and provider-createdAt/daily release are retrospective scenarios.",
            "No authentic G0, accepted execution or all-state production availability; no live acquisition or deployment.",
            "24h scenario does not certify current300s transfer; September is partial; only3assessmentmonthly blocks.",
            "q includes the execution book; leave-execution-book-out with>=2peers is diagnostic only."]}
    _save(stage05/"protocol.json", protocol)
    _save(stage05/"freeze_receipt.json", {"protocol_file_sha256": _sha(stage05/"protocol.json"),
        "protocol_sha256": _canonical_sha(protocol), "formal_head_fits": 0, "base_fits": 0})
    print(f"FROZEN {stage05/'protocol.json'} canonical_sha256={_canonical_sha(protocol)}")
    return stage05/"protocol.json"


def _verify(protocol, root):
    expected = {"schema": SCHEMA, "scope": "EXPOSED_HISTORICAL_DEVELOPMENT_ONLY",
        "native_identity": f"Causal-C0:{C0_NATIVE_VERSION}", "mode": "full", "epsilon": C0_EPSILON,
        "calendar": {name: list(dates) for name, dates in PHASES.items()},
        "lambdas": list(LAMBDAS), "recipes": list(RECIPES), "formal_fit_budget": 18,
        "base_fit_budget": 0, "max_quote_age_seconds": 86400, "minimum_books": 2,
        "tax": .12, "minimum_net_ev_exclusive": .05}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Changed historical development contract")
    for reference in protocol["sources"].values():
        if _sha(_source_path(reference, root)) != reference["sha256"]:
            raise ValueError(f"Changed frozen input: {reference['path']}")
    for path, digest in {**protocol["code_hashes"], **protocol["preserved_forward_code_hashes"]}.items():
        if _sha(PROJECT_ROOT/path) != digest:
            raise ValueError(f"Changed frozen implementation: {path}")


def _load(protocol, root):
    sources = protocol["sources"]
    control = pd.read_parquet(_source_path(sources["control"], root))
    market = pd.read_parquet(_source_path(sources["market"], root))
    quotes = pd.read_parquet(_source_path(sources["quotes"], root))
    with _source_path(sources["availability"], root).open() as stream:
        availability = {str(row["golgg_match_id"]): row["label_release_day"]
            for row in (json.loads(line) for line in stream)}
    release_map = json.loads(_source_path(sources["release_map"], root).read_text())
    effective = []
    for _, row in control.iterrows():
        key = str(row.golgg_match_id)
        if key not in availability:
            raise ValueError("Native label release absent from bound bank")
        candidates = [pd.Timestamp(row.result_day), pd.Timestamp(availability[key])]
        if key in release_map:
            candidates.append(pd.Timestamp(release_map[key]))
        if any(pd.isna(value) for value in candidates):
            raise ValueError("Invalid bound daily release")
        effective.append(max(candidates).date().isoformat())
    control["effective_label_release_day"] = effective
    return control, market, quotes


def run(protocol_path, research_root=None):
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    root = _resolve_research_root(research_root)
    seal = json.loads((protocol_path.parent/"freeze_receipt.json").read_text())
    if (seal["protocol_file_sha256"] != _sha(protocol_path)
            or seal["protocol_sha256"] != _canonical_sha(protocol)):
        raise ValueError("Frozen historical protocol seal differs; never repair its hash")
    _verify(protocol, root)
    run_id = protocol["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("Invalid run ID")
    stage05 = PROJECT_ROOT/"data/05_model_input/c0_betting_development"/run_id
    stage06 = PROJECT_ROOT/"data/06_models/c0_betting_development"/run_id
    stage07 = PROJECT_ROOT/"data/07_model_output/c0_betting_development"/run_id
    stage08 = PROJECT_ROOT/"data/08_reporting/c0_betting_development"/run_id
    if stage06.exists() or stage07.exists() or (stage08/"decision.json").exists():
        raise FileExistsError("Training/output already exists; no adaptive retry or overwrite")
    control, market, quotes = _load(protocol, root)
    panel, coverage = build_panel(control, market, quotes)
    counts = protocol["expected_population"]
    if (len(control) != counts["native"] or len(panel) != counts["provider"]
            or panel.market_eligible.sum() != counts["primary_common"]):
        raise ValueError("Frozen population mismatch")
    stage05.mkdir(parents=True, exist_ok=True)
    stage06.mkdir(parents=True, exist_ok=False)
    stage07.mkdir(parents=True, exist_ok=False)
    stage08.mkdir(parents=True, exist_ok=True)
    for phase in PHASES:
        panel.loc[panel.phase == phase].to_parquet(stage05/f"{phase.lower()}_denominator.parquet", index=False)
    panel.to_parquet(stage05/"all_provider_denominator.parquet", index=False)
    control.to_parquet(stage05/"all_native_coverage.parquet", index=False)
    _save(stage05/"coverage.json", coverage)
    train = panel.loc[(panel.phase == "TRAIN") & panel.fit_eligible & panel.market_eligible].copy()
    cal = panel.loc[(panel.phase == "CAL") & panel.fit_eligible & panel.market_eligible].copy()
    bindings = {"protocol_sha256": _canonical_sha(protocol), "protocol_file_sha256": _sha(protocol_path),
        "scope": "historical_development_only", "train_row_ids_sha256": _canonical_sha(train.row_id.tolist()),
        "sources": {name: reference["sha256"] for name, reference in protocol["sources"].items()},
        "hypothetical_fit_origin": "2026-05-01", "hypothetical_cal_selection_origin": "2026-07-01",
        "actual_fit_started_at_utc": datetime.now(timezone.utc).isoformat()}
    train.attrs["c0_bindings"] = bindings
    heads, selections, fit_count = {}, {}, 0
    for recipe in RECIPES:
        if recipe in {"S", "M", "Fixed50"}:
            head = fit_c0_market_head(recipe, train, regularization=0.)
            selection = {"fixed": True, "lambda": 0.}
        else:
            candidates = []
            for regularization in LAMBDAS:
                fitter = fit_price_head if recipe in {"M-price", "H-price"} else fit_c0_market_head
                candidate = fitter(recipe, train, regularization=regularization)
                fit_count += 1
                _save(stage06/f"{recipe}_lambda_{regularization:g}.json", candidate.to_dict())
                candidates.append(candidate)
            head, selection = select_c0_market_head(candidates, cal)
        heads[recipe] = head
        selections[recipe] = selection
        _save(stage06/f"{recipe}_selected.json", head.to_dict())
    if fit_count != protocol["formal_fit_budget"]:
        raise RuntimeError("Wrong formal fit budget")
    _save(stage06/"selection.json", {"actual_selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": selections, "formal_head_fits": fit_count, "base_fits": 0, "bindings": bindings})
    _save(stage05/"fit_membership.json", {"TRAIN": train.row_id.tolist(), "CAL": cal.row_id.tolist(),
        "ASSESS": panel.loc[panel.phase == "ASSESS", "row_id"].tolist(), "bindings": bindings})
    assess = panel.loc[panel.phase == "ASSESS"].copy().reset_index(drop=True)
    assess.attrs["monthly_universe"] = sorted(assess.month.unique())
    result = evaluate_development(assess, heads, draws=protocol["statistics"]["bootstrap_draws"],
        seed=protocol["statistics"]["seed"])
    pd.DataFrame(result["rows"]).to_parquet(stage07/"assessment_predictions_and_ledger.parquet", index=False)
    _save(stage08/"assessment.json", {key: value for key, value in result.items() if key != "rows"})
    fresh, fresh_coverage = build_panel(control, market, quotes, max_age_seconds=300)
    fresh_assess = fresh.loc[fresh.phase == "ASSESS"].copy().reset_index(drop=True)
    fresh_assess.attrs["monthly_universe"] = sorted(assess.month.unique())
    diagnostic = evaluate_development(fresh_assess, heads, draws=protocol["statistics"]["bootstrap_draws"],
        seed=protocol["statistics"]["seed"])
    pd.DataFrame(diagnostic["rows"]).to_parquet(stage07/"fresh_300s_diagnostic_predictions_and_ledger.parquet", index=False)
    _save(stage08/"fresh_300s_diagnostic.json", {"classification": "TRANSFER_DIAGNOSTIC_NO_FIT_NO_SELECTION",
        "coverage": fresh_coverage, "result": {key: value for key, value in diagnostic.items() if key != "rows"}})
    decision = {"schema": SCHEMA, "status": "COMPLETED", "decision": result["decision"],
        "run_id": run_id, "scope": protocol["scope"], "actual_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_head_fits": fit_count, "base_fits": 0, "coverage": coverage, "gates": result.get("gates"),
        "protocol_sha256": _canonical_sha(protocol), "protocol_file_sha256": _sha(protocol_path),
        "canonical_models_changed": False, "deployed": False, "live_quote_requests": 0,
        "forward_study_status": "UNCHANGED_G0_BLOCKED_NOT_RUN", "limitations": protocol["limitations"]}
    _save(stage08/"decision.json", decision)
    outputs = [path for directory in (stage05, stage06, stage07, stage08)
        for path in directory.iterdir() if path.is_file()]
    _save(stage08/"execution_receipt.json", {"decision": decision,
        "artifacts": {str(path.relative_to(PROJECT_ROOT)): {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in sorted(outputs)}, "preserved_forward_code_hashes": protocol["preserved_forward_code_hashes"]})
    print(json.dumps({"run_id": run_id, "decision": decision["decision"], "formal_head_fits": fit_count,
        "assessment_common": result["metrics"]["H-price"]["common_count"],
        "H_price_LL": result["metrics"]["H-price"]["LL"], "report": str(stage08/"assessment.json")}, sort_keys=True))
    return decision


def reassess(protocol_path, execution_binding, research_root=None):
    """Repair assessment accounting on existing heads; never invoke a fitter."""
    protocol_path = Path(protocol_path).resolve()
    binding_path = Path(execution_binding).resolve()
    protocol = json.loads(protocol_path.read_text())
    binding = json.loads(binding_path.read_text())
    if (binding.get("kind") != "assessment_only_cash_accounting_review"
            or binding.get("protocol_sha256") != _canonical_sha(protocol)
            or binding.get("protocol_file_sha256") != _sha(protocol_path)
            or binding.get("additional_formal_fits") != 0):
        raise ValueError("Wrong reviewed execution binding")
    output_stages = {stage: PROJECT_ROOT/"data"/stage/"c0_betting_development"/protocol["run_id"]
        for stage in ("05_model_input", "07_model_output", "08_reporting")}
    reviewed_outputs = [output_stages["08_reporting"]/"reviewed_decision.json"]
    for prefix in ("reviewed_assessment", "reviewed_fresh_300s_diagnostic"):
        reviewed_outputs.extend((output_stages["05_model_input"]/f"{prefix}_denominator.parquet",
            output_stages["07_model_output"]/f"{prefix}_predictions_and_ledger.parquet",
            output_stages["08_reporting"]/f"{prefix}.json"))
    if any(path.exists() for path in reviewed_outputs):
        raise FileExistsError("Reviewed outputs are immutable; reassessment cannot overwrite an existing run")
    seal = json.loads((protocol_path.parent/"freeze_receipt.json").read_text())
    if seal["protocol_file_sha256"] != _sha(protocol_path) or seal["protocol_sha256"] != _canonical_sha(protocol):
        raise ValueError("Original frozen protocol changed")
    if (set(binding["reviewed_code_hashes"]) != set(CODE)
            or set(binding["original_frozen_code"]) != set(CODE)
            or set(protocol["code_hashes"]) != set(CODE)):
        raise ValueError("Reviewed implementation closure is incomplete")
    root = _resolve_research_root(research_root)
    reviewed_verification = {**protocol, "code_hashes": binding["reviewed_code_hashes"]}
    _verify(reviewed_verification, root)
    for path, reference in binding["original_frozen_code"].items():
        if (reference["sha256"] != protocol["code_hashes"][path]
                or _sha(PROJECT_ROOT/reference["snapshot_path"]) != reference["sha256"]):
            raise ValueError("Original training source snapshot changed")
    for path, digest in binding["original_model_hashes"].items():
        if _sha(PROJECT_ROOT/path) != digest:
            raise ValueError("A saved head/selection changed; reassessment cannot refit")
    run_id = protocol["run_id"]
    stage05 = PROJECT_ROOT/"data/05_model_input/c0_betting_development"/run_id
    stage06 = PROJECT_ROOT/"data/06_models/c0_betting_development"/run_id
    stage07 = PROJECT_ROOT/"data/07_model_output/c0_betting_development"/run_id
    stage08 = PROJECT_ROOT/"data/08_reporting/c0_betting_development"/run_id
    expected_models = {str((stage06/f"{recipe}_selected.json").relative_to(PROJECT_ROOT)) for recipe in RECIPES}
    expected_models |= {str((stage06/f"{recipe}_lambda_{value:g}.json").relative_to(PROJECT_ROOT))
        for recipe in RECIPES if recipe not in {"S", "M", "Fixed50"} for value in LAMBDAS}
    expected_models.add(str((stage06/"selection.json").relative_to(PROJECT_ROOT)))
    if set(binding["original_model_hashes"]) != expected_models:
        raise ValueError("Reviewed saved-model closure is incomplete")
    control, market, quotes = _load(protocol, root)
    panel, coverage = build_panel(control, market, quotes)
    membership = json.loads((stage05/"fit_membership.json").read_text())
    for phase in ("TRAIN", "CAL"):
        current = panel.loc[(panel.phase == phase) & panel.fit_eligible & panel.market_eligible, "row_id"].tolist()
        if current != membership[phase]:
            raise ValueError("Reviewed accounting changes a frozen fit/selection population")
    original = pd.read_parquet(stage05/"all_provider_denominator.parquet").set_index("row_id")
    current = panel.set_index("row_id").reindex(original.index)
    for column in ("p_C0", "q", "odds_a", "odds_b", "y"):
        if not np.array_equal(current[column].to_numpy(), original[column].to_numpy(), equal_nan=True):
            raise ValueError("Reviewed accounting changes frozen prediction/price/label inputs")
    heads = {}
    for recipe in RECIPES:
        value = json.loads((stage06/f"{recipe}_selected.json").read_text())
        heads[recipe] = (C0PriceHead if recipe in {"M-price", "H-price"} else C0MarketHead).from_dict(value)
    results = {}
    for age, prefix in ((86400, "reviewed_assessment"), (300, "reviewed_fresh_300s_diagnostic")):
        all_rows, age_coverage = (panel, coverage) if age == 86400 else build_panel(
            control, market, quotes, max_age_seconds=age)
        assess = all_rows.loc[all_rows.phase == "ASSESS"].copy().reset_index(drop=True)
        assess.attrs["monthly_universe"] = sorted(panel.loc[panel.phase == "ASSESS", "month"].unique())
        if len(assess) != 523 or int(assess.financial_eligible.sum()) != 523:
            raise ValueError("Reviewed cash population is not all523matureopportunities")
        result = evaluate_development(assess, heads, draws=5000, seed=20261006)
        assess.to_parquet(stage05/f"{prefix}_denominator.parquet", index=False)
        pd.DataFrame(result["rows"]).to_parquet(stage07/f"{prefix}_predictions_and_ledger.parquet", index=False)
        summary = {key: value for key, value in result.items() if key != "rows"}
        _save(stage08/f"{prefix}.json", summary if age == 86400 else {
            "classification": "TRANSFER_DIAGNOSTIC_NO_FIT_NO_SELECTION", "coverage": age_coverage, "result": summary})
        results[age] = result
    for path, digest in binding["original_model_hashes"].items():
        if _sha(PROJECT_ROOT/path) != digest:
            raise RuntimeError("Reassessment mutated a frozen model")
    decision = {"schema": SCHEMA, "status": "COMPLETED_REVIEWED_ASSESSMENT", "decision": results[86400]["decision"],
        "scope": protocol["scope"], "run_id": run_id, "formal_head_fits": 18, "additional_formal_fits": 0,
        "base_fits": 0, "gates": results[86400]["gates"], "coverage": coverage,
        "cash_denominator": 523, "primary_common": 514, "fresh_diagnostic_common": 57,
        "protocol_sha256": _canonical_sha(protocol), "protocol_file_sha256": _sha(protocol_path),
        "reviewed_execution_binding_sha256": _sha(binding_path),
        "supersedes_for_current_analysis": "Pre-review assessment.json/decision.json remain unchanged provenance.",
        "original_model_hashes_unchanged": True, "forward_code_hashes_unchanged": True,
        "canonical_models_changed": False, "deployed": False, "live_quote_requests": 0,
        "forward_study_status": "UNCHANGED_G0_BLOCKED_NOT_RUN", "limitations": protocol["limitations"],
        "actual_completed_at_utc": datetime.now(timezone.utc).isoformat()}
    _save(stage08/"reviewed_decision.json", decision)
    print(json.dumps({"decision": decision["decision"], "retained_formal_fits": 18, "new_fits": 0,
        "cash_denominator": 523, "H_price_LL": results[86400]["metrics"]["H-price"]["LL"]}, sort_keys=True))
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(prog="run_unified_betting_evaluation.py approach-comparison c0-historical-development")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--reassess", action="store_true")
    parser.add_argument("--execution-binding", type=Path)
    args = parser.parse_args(argv)
    if args.freeze:
        if not args.run_id or args.protocol or args.reassess or args.execution_binding:
            parser.error("--freeze requires only --run-id and optional --research-root")
        return freeze(args.run_id, args.research_root)
    if args.reassess:
        if not args.protocol or not args.execution_binding or args.run_id:
            parser.error("--reassess requires --protocol/--execution-binding and forbids --run-id")
        return reassess(args.protocol, args.execution_binding, args.research_root)
    if args.execution_binding:
        parser.error("--execution-binding is only valid with --reassess")
    if not args.protocol or args.run_id:
        parser.error("training requires --protocol and forbids --run-id")
    return run(args.protocol, args.research_root)
