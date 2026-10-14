#!/usr/bin/env python3
"""Unified CLI runner for LoL tournament distribution and bracket simulation benchmark.

Bridges prior-CAL fitted Glicko-format, fixed regional proxies and Uniform to the
multistage tournament simulation engine across historical international editions:
- worlds-2023 (Swiss + Single Elimination, 16 teams)
- worlds-2024 (Swiss + Single Elimination, 16 teams)
- msi-2023 (Double Elimination Bracket, 8 teams)
- msi-2024 (Double Elimination Bracket, 8 teams)

Usage examples:
  python scripts/run_tournament_distribution_benchmark.py --freeze-manifest  # read-only validation
  python scripts/run_tournament_distribution_benchmark.py --smoke
  python scripts/run_tournament_distribution_benchmark.py --benchmark --run-id new_proxy_diagnostic
  python scripts/run_tournament_distribution_benchmark.py --simulate --edition worlds-2023 --model regional_glicko_proxy
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from numbers import Real
from typing import Any

import numpy as np

# Ensure repository root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.tournament_metrics import (
    evaluate_tournament_edition,
    extract_tournament_ground_truth,
)
from src.models.tournament_bracket_simulator import TournamentBracketSimulator
from src.models.tournament_catalog import instantiate_profile
from src.models.tournament_pair_adapter import MODEL_KINDS, TournamentPairAdapter, validate_model_kind


DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "conf/base/tournament_manifest.json"
DEFAULT_MODELS = MODEL_KINDS
DEFAULT_EDITIONS = ("worlds-2023", "worlds-2024", "msi-2023", "msi-2024")
DIAGNOSTIC_TARGET = "calibrated_regional_glicko_proxy"
DIAGNOSTIC_COMPARATORS = ("uniform", "glicko_format")
DIAGNOSTIC_METRICS = {
    "champion_brier": ("mean_champion_brier", "champion_brier_edition_count"),
    "swiss_rps": ("mean_swiss_record_rps", "swiss_record_rps_edition_count"),
}


def proxy_decision_contract() -> dict[str, Any]:
    """Declared before scoring; this diagnostic cannot establish native adoption."""
    return {
        "schema_version": 1,
        "target": DIAGNOSTIC_TARGET,
        "comparators": list(DIAGNOSTIC_COMPARATORS),
        "metrics": {metric: list(fields) for metric, fields in DIAGNOSTIC_METRICS.items()},
        "comparison": "strictly lower for both metrics against both comparators",
        "support": "positive integer edition counts, matched across every declared model",
        "missingness": "missing or nonfinite required scores fail; no reduced intersection",
        "flow": "verified true for every declared and supplied model",
        "uncertainty": "edition-level generalization uncertainty not established by this proxy diagnostic",
        "native_adoption_eligible": False,
    }


def _finite_score(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def evaluate_proxy_decision(
    model_aggregates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Require all predictive/support/flow gates, without ever promoting a native model."""
    criteria: dict[str, bool] = {}
    estimable = all(model in model_aggregates for model in MODEL_KINDS)
    for metric, (score_key, support_key) in DIAGNOSTIC_METRICS.items():
        rows = [model_aggregates.get(model, {}) for model in MODEL_KINDS]
        supports = [row.get(support_key) for row in rows]
        support_valid = all(type(count) is int and count > 0 for count in supports)
        support_matched = support_valid and len(set(supports)) == 1
        metric_estimable = support_matched and all(_finite_score(row.get(score_key)) for row in rows)
        criteria[f"{metric}_support_matched"] = bool(support_matched)
        estimable = estimable and metric_estimable
        target_score = model_aggregates.get(DIAGNOSTIC_TARGET, {}).get(score_key)
        for comparator in DIAGNOSTIC_COMPARATORS:
            comparator_score = model_aggregates.get(comparator, {}).get(score_key)
            criteria[f"{DIAGNOSTIC_TARGET}_beats_{comparator}_{metric}"] = bool(
                metric_estimable and target_score < comparator_score
            )
    criteria["comparisons_estimable"] = bool(estimable)
    flow_pass = (
        all(model in model_aggregates for model in MODEL_KINDS)
        and all(row.get("all_flow_feasible") is True for row in model_aggregates.values())
    )
    criteria["network_flow_feasibility"] = flow_pass
    return {
        "decision": "PROXY_DIAGNOSTIC_COMPLETED" if all(criteria.values()) else "PROXY_DIAGNOSTIC_FAILED",
        "native_adoption_eligible": False,
        "flow_conservation_all_pass": flow_pass,
        "criteria": criteria,
        "decision_contract": proxy_decision_contract(),
    }


def compute_sha256(path: Path | str) -> str:
    """Compute hex SHA-256 digest of file."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    """Load and validate tournament manifest."""
    manifest_path = manifest_path if manifest_path is not None else DEFAULT_MANIFEST_PATH
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Tournament manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema_version: {data.get('schema_version')}")
    return data


def freeze_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    """Validate the already frozen manifest and sources without rewriting any bytes."""
    manifest_path = manifest_path if manifest_path is not None else DEFAULT_MANIFEST_PATH
    print(f"--> Validating tournament manifest at {manifest_path}...")
    manifest = load_manifest(manifest_path)

    for src in manifest.get("sources", []):
        src_path = PROJECT_ROOT / src["path"]
        actual_sha = compute_sha256(src_path)
        expected_sha = src.get("sha256")
        if actual_sha != expected_sha:
            raise ValueError(
                f"Source hash mismatch for {src['path']}: expected {expected_sha}, got {actual_sha}"
            )
        print(f"  [OK] Source verified: {src['path']} ({actual_sha[:12]}...)")

    for eid, ed_obj in manifest.get("editions", {}).items():
        entrants = ed_obj["entrants"]
        if len(entrants) != ed_obj["team_count"]:
            raise ValueError(f"Edition {eid}: entrant count {len(entrants)} != {ed_obj['team_count']}")
        print(f"  [OK] Edition verified: {eid} ({ed_obj['name']}, {len(entrants)} teams)")

    print(f"==> Frozen tournament manifest validated (unchanged): {manifest_path}")
    return manifest


def run_simulation(
    edition_id: str,
    model_kind: str,
    manifest: dict[str, Any],
    adapter: TournamentPairAdapter,
    simulations: int = 10000,
    seed: int = 20261005,
) -> dict[str, Any]:
    """Simulate one diagnostic model with its loaded-source/fitted-parameter receipt."""
    validate_model_kind(model_kind)
    ed_info = manifest["editions"][edition_id]
    profile_id = ed_info["profile"]
    teams = ed_info["entrants"]
    cutoff = ed_info["cutoff"]

    spec = instantiate_profile(profile_id, teams)
    probs = adapter.predict_pair_table(teams, cutoff, model_kind=model_kind)
    predictor_receipt = adapter.predictor_receipt(teams, cutoff, model_kind=model_kind)

    simulator = TournamentBracketSimulator(simulations=simulations, seed=seed)
    result = simulator.simulate(spec, probs)
    result["edition_id"] = edition_id
    result["model_kind"] = model_kind
    result["cutoff"] = cutoff
    result["predictor_receipt"] = predictor_receipt
    return result


def run_smoke_test() -> None:
    """Run fast 100-simulation verification across all 4 editions and 4 models."""
    print("=" * 70)
    print("Running Tournament Distribution Smoke Test (100 simulations per edition/model)")
    print("=" * 70)

    manifest = load_manifest()
    adapter = TournamentPairAdapter()
    smoke_sims = 100

    start_time = time.perf_counter()
    rows = []

    for eid in DEFAULT_EDITIONS:
        ed_info = manifest["editions"][eid]
        gt = extract_tournament_ground_truth(ed_info["event_title"])

        for model in DEFAULT_MODELS:
            res = run_simulation(eid, model, manifest, adapter, simulations=smoke_sims, seed=42)
            eval_res = evaluate_tournament_edition(eid, res, gt)
            rows.append({
                "edition": eid,
                "model": model,
                "champ_brier": eval_res["champion_brier"],
                "champ_ll": eval_res["champion_log_loss"],
                "final_brier": eval_res["finalists_brier"],
                "swiss_rps": eval_res.get("swiss_record_rps"),
                "flow_feasible": eval_res["network_flow_feasible"],
            })
            print(
                f"[{eid:12s} | {model:14s}] Champ BS: {eval_res['champion_brier']:.4f} | "
                f"Final BS: {eval_res['finalists_brier']:.4f} | "
                f"Flow Feasible: {eval_res['network_flow_feasible']}"
            )

    elapsed = time.perf_counter() - start_time
    print("-" * 70)
    print(f"Smoke test complete in {elapsed:.2f}s across {len(rows)} simulations.")
    assert all(r["flow_feasible"] for r in rows), "Network flow conservation violated!"
    print("ALL 16 SMOKE RUNS PASSED AND PRESERVED FLOW CONSERVATION.")


def run_benchmark(
    run_id: str,
    simulations: int = 10000,
    seed: int = 20261005,
    output_base: Path | None = None,
) -> dict[str, Any]:
    """Persist full forecasts and strict JSON reports for the proxy benchmark.

    Model/scorer losses remain mathematical and unclipped, including infinity
    when an observed outcome has zero probability. Only the report boundary
    converts nonfinite values to null with explicit ``nonfinite_reasons``.
    The returned decision is the same strict report object saved on disk.
    """
    from src.models.tournament_improvement_runner import save_json

    print("=" * 78)
    print(f"STARTING TOURNAMENT DISTRIBUTION BENCHMARK: {run_id}")
    print(f"Simulations: {simulations:,} | Seed: {seed}")
    print("=" * 78)

    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a nonempty directory name")
    rep_dir = (output_base or PROJECT_ROOT / "data/08_reporting/tournament_distribution") / run_id
    out_dir = (PROJECT_ROOT / "data/07_model_output/tournament_distribution") / run_id
    if rep_dir.exists() or out_dir.exists():
        raise FileExistsError(f"Run outputs already exist; choose a new run_id: {run_id}")
    manifest = freeze_manifest()
    adapter = TournamentPairAdapter()
    rep_dir.mkdir(parents=True, exist_ok=False)
    out_dir.mkdir(parents=True, exist_ok=False)
    contract = {
        **proxy_decision_contract(),
        "run_id": run_id,
        "editions": list(DEFAULT_EDITIONS),
        "models": list(DEFAULT_MODELS),
        "simulations": simulations,
        "seed": seed,
        "manifest_sha256": compute_sha256(DEFAULT_MANIFEST_PATH),
    }
    save_json(rep_dir / "protocol.json", contract)

    all_scores: list[dict[str, Any]] = []
    predictor_receipts: dict[str, Any] = {}
    swiss_editions: set[str] = set()
    start_total = time.perf_counter()

    for eid in DEFAULT_EDITIONS:
        ed_info = manifest["editions"][eid]
        gt = extract_tournament_ground_truth(ed_info["event_title"])
        if gt.swiss_records:
            swiss_editions.add(eid)
        print(f"\n--- Simulating Edition: {eid} ({ed_info['name']}) ---")
        print(f"Entrants ({len(ed_info['entrants'])}): {', '.join(ed_info['entrants'])}")
        print(f"Ground truth: Champion={gt.champion}, Finalists={gt.finalists}")

        predictor_receipts[eid] = {}

        for model in DEFAULT_MODELS:
            print(f"  -> Running {model} ({simulations:,} draws)...", end="", flush=True)
            res = run_simulation(eid, model, manifest, adapter, simulations=simulations, seed=seed)
            eval_res = evaluate_tournament_edition(eid, res, gt)
            predictor_receipts[eid][model] = res["predictor_receipt"]
            eval_res["model"] = model
            all_scores.append(eval_res)

            # Save model distribution artifact
            dist_file = out_dir / f"{eid}_{model}_distributions.json"
            save_json(dist_file, res)

            print(f" Done. (Champ BS: {eval_res['champion_brier']:.4f}, Final BS: {eval_res['finalists_brier']:.4f})")

    elapsed_total = time.perf_counter() - start_total

    # Save CSV of edition scores
    csv_path = rep_dir / "edition_scores.csv"
    fieldnames = [
        "edition_id", "model", "actual_champion", "champion_prob",
        "champion_log_loss", "champion_brier", "finalists_brier",
        "advance_brier", "swiss_record_rps", "playoff_series_length_rps",
        "playoff_scoreline_log_loss", "network_flow_feasible", "max_flow_violation"
    ]
    with csv_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_scores:
            writer.writerow(row)

    # Compute aggregate metrics across models
    model_aggregates: dict[str, dict[str, Any]] = {}
    for model in DEFAULT_MODELS:
        m_rows = [r for r in all_scores if r["model"] == model]
        champion_rows = [r["champion_brier"] for r in m_rows if _finite_score(r.get("champion_brier"))]
        champ_bs = float(np.mean(champion_rows)) if champion_rows and len(champion_rows) == len(m_rows) else None
        champ_ll = float(np.mean([r["champion_log_loss"] for r in m_rows]))
        final_bs = float(np.mean([r["finalists_brier"] for r in m_rows]))

        adv_rows = [r["advance_brier"] for r in m_rows if r.get("advance_brier") is not None]
        swiss_adv_bs = float(np.mean(adv_rows)) if adv_rows else None

        expected_swiss_rows = [r for r in m_rows if r["edition_id"] in swiss_editions]
        rps_rows = [r["swiss_record_rps"] for r in expected_swiss_rows if _finite_score(r.get("swiss_record_rps"))]
        swiss_rps = float(np.mean(rps_rows)) if rps_rows and len(rps_rows) == len(expected_swiss_rows) else None

        all_flow = bool(m_rows) and all(r.get("network_flow_feasible") is True for r in m_rows)

        model_aggregates[model] = {
            "mean_champion_brier": round(champ_bs, 6) if champ_bs is not None else None,
            "mean_champion_log_loss": round(champ_ll, 6),
            "mean_finalists_brier": round(final_bs, 6),
            "mean_swiss_advance_brier": round(swiss_adv_bs, 6) if swiss_adv_bs is not None else None,
            "mean_swiss_record_rps": round(swiss_rps, 6) if swiss_rps is not None else None,
            "champion_brier_edition_count": len(champion_rows),
            "swiss_record_rps_edition_count": len(rps_rows),
            "all_flow_feasible": all_flow,
        }

    decision_payload = {
        **evaluate_proxy_decision(model_aggregates),
        "run_id": run_id,
        "status": "COMPLETED",
        "model_comparison": model_aggregates,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(rep_dir / "decision.json", decision_payload)

    # Metrics JSON
    metrics_payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_simulations": len(all_scores) * simulations,
        "elapsed_seconds": round(elapsed_total, 2),
        "editions": DEFAULT_EDITIONS,
        "models": DEFAULT_MODELS,
        "model_aggregates": model_aggregates,
        "detailed_scores": all_scores,
    }
    save_json(rep_dir / "metrics.json", metrics_payload)

    # Receipt JSON
    receipt_payload = {
        "run_id": run_id,
        "status": "COMPLETED",
        "manifest_path": str(DEFAULT_MANIFEST_PATH),
        "manifest_sha256": compute_sha256(DEFAULT_MANIFEST_PATH),
        "sources": manifest.get("sources", []),
        "decision_contract": contract,
        "predictor_receipts": predictor_receipts,
        "outputs": {
            "protocol": str(rep_dir / "protocol.json"),
            "metrics": str(rep_dir / "metrics.json"),
            "decision": str(rep_dir / "decision.json"),
            "edition_scores_csv": str(rep_dir / "edition_scores.csv"),
            "model_outputs_dir": str(out_dir),
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(rep_dir / "receipt.json", receipt_payload)

    # Print summary table
    print("\n" + "=" * 78)
    print("TOURNAMENT DISTRIBUTION BENCHMARK SUMMARY")
    print("=" * 78)
    print(f"{'Model':<16} | {'Champ BS':<10} | {'Champ LL':<10} | {'Final BS':<10} | {'Swiss RPS':<10} | {'Flow OK':<8}")
    print("-" * 78)
    for model in DEFAULT_MODELS:
        agg = model_aggregates[model]
        s_rps = f"{agg['mean_swiss_record_rps']:.4f}" if agg["mean_swiss_record_rps"] is not None else "N/A"
        champ_bs_text = f"{agg['mean_champion_brier']:.4f}" if agg["mean_champion_brier"] is not None else "N/A"
        print(
            f"{model:<16} | {champ_bs_text:<10} | "
            f"{agg['mean_champion_log_loss']:<10.4f} | {agg['mean_finalists_brier']:<10.4f} | "
            f"{s_rps:<10} | {str(agg['all_flow_feasible']):<8}"
        )
    print("=" * 78)
    print(f"Decision: {decision_payload['decision']} (native adoption ineligible)")
    print(f"Artifacts preserved in: {rep_dir}")
    print("=" * 78)

    return json.loads((rep_dir / "decision.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical proxy diagnostics and frozen native C0 research study")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--freeze-manifest", action="store_true", help="Read-only validation of the frozen proxy manifest")
    actions.add_argument("--smoke", action="store_true", help="Run the historical proxy smoke")
    actions.add_argument("--benchmark", action="store_true", help="Proxy benchmark, or frozen native study with --native-study")
    actions.add_argument("--simulate", action="store_true", help="Write a new forecast artifact")
    actions.add_argument("--score", action="store_true", help="Load --forecast and score its actual probabilities")
    actions.add_argument("--freeze-study", action="store_true", help="Freeze native study sources, memberships, code and artifacts before fitting")
    actions.add_argument("--run-study", action="store_true", help="Fit/select/predict/simulate/score the frozen native study")
    actions.add_argument("--study-smoke", action="store_true", help="Real native/filter/score-law nonformal smoke in a separate namespace")
    actions.add_argument("--numerical-benchmark", action="store_true", help="New numerical integration comparison using retained native scenario inputs; no fitting or promotion")
    actions.add_argument("--numerical-smoke", action="store_true", help="Retained-input exact/hybrid smoke; eight prefixes by default, no persistent artifacts")
    actions.add_argument("--freeze-copula", action="store_true", help="Freeze the single marginal-preserving C0 tournament copula recipe and source contracts")
    actions.add_argument("--run-copula", action="store_true", help="Fit and evaluate the frozen C0 copula study without changing native artifacts")
    actions.add_argument("--copula-smoke", action="store_true", help="Exercise the frozen copula inputs in an isolated nonformal smoke namespace")
    actions.add_argument("--prepare-native-2023", action="store_true", help="Explicit isolated exact W32/E12 2023 control preparation; never an A0 alias")
    parser.add_argument("--native-study", action="store_true", help="Use frozen native policy for --simulate/--score/--benchmark; --model stays proxy-only")
    parser.add_argument("--protocol", type=Path, help="Frozen native study protocol.json")
    parser.add_argument("--research-root", type=Path, help="Local archive root; otherwise resolve existing environment/root helper")
    parser.add_argument("--edition", type=str, help="Frozen event/edition identifier (proxy commands accept only the four historical editions)")
    parser.add_argument("--model", type=str, choices=DEFAULT_MODELS, help="Diagnostic model: prior-CAL fitted glicko_format, fixed regional proxies or uniform; never native c0/a0")
    parser.add_argument("--study-model", choices=("native_c0_unchanged", "residual_posterior_mean", "residual_independent_matched_marginals", "residual_coherent", "origin_eligible_annual_a0", "prior_calibrated_glicko_format", "uniform"), help="Origin-eligible native study arm/comparator")
    parser.add_argument("--source-run-id", help="Retained native study providing numerical-comparison inputs")
    parser.add_argument("--integrations", nargs="+", choices=("monte_carlo", "exact", "hybrid"), help="Methods for --numerical-benchmark (default: monte_carlo hybrid)")
    parser.add_argument("--simulations", type=int, help="Paths/prefixes: default 10000, or 8 for --numerical-smoke")
    parser.add_argument("--seed", type=int, help="Proxy/numerical RNG seed (default 20261005); fitting studies fix their own seed")
    parser.add_argument("--run-id", type=str, default=None, help="Exclusive run namespace")
    parser.add_argument("--output-dir", type=Path, help="Proxy benchmark output directory")
    parser.add_argument("--output", type=Path, help="New single forecast or score artifact (never overwrite)")
    parser.add_argument("--forecast", type=Path, help="Existing forecast JSON to score")
    args = parser.parse_args()
    requested_simulations = args.simulations
    requested_seed = args.seed
    if args.seed is None:
        args.seed = 20261005
    if args.simulations is None:
        args.simulations = 10000
    run_id = args.run_id or f"tournament_dist_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if args.freeze_copula or args.run_copula or args.copula_smoke:
        from src.models.tournament_copula_runner import freeze_copula_study, run_copula_study
        from src.models.tournament_improvement_runner import _safe
        if (args.native_study or args.model or args.study_model or args.edition
                or args.output_dir or args.output or args.forecast or args.source_run_id
                or args.integrations or requested_simulations is not None or requested_seed is not None):
            parser.error("Copula commands use the frozen full cohort, controls and numerical budgets; overrides are not supported")
        if args.freeze_copula:
            if not args.run_id or args.protocol:
                parser.error("--freeze-copula requires a new --run-id and no --protocol")
            print(freeze_copula_study(args.run_id, research_root=args.research_root))
        else:
            if args.protocol is None or args.run_id:
                parser.error("--run-copula/--copula-smoke require --protocol and its frozen namespace; --research-root may locate the same bound bytes")
            print(json.dumps(_safe(run_copula_study(
                args.protocol, research_root=args.research_root, smoke=args.copula_smoke)), indent=2))
        return
    if args.numerical_benchmark or args.numerical_smoke:
        from src.models.tournament_numerical_benchmark import run_numerical_benchmark, numerical_smoke
        from src.models.tournament_improvement_runner import _safe
        if (args.native_study or args.protocol or args.model or args.output_dir
                or args.output or args.forecast or args.research_root):
            parser.error("Numerical commands use retained native --source-run-id inputs, not proxy/model-fitting options")
        source_run_id = args.source_run_id or "tournament_c0_20261006_01"
        if args.numerical_smoke:
            if args.integrations or args.edition or args.study_model:
                parser.error("--numerical-smoke exercises the fixed four-edition, four-model smoke contract")
            result = numerical_smoke(
                source_run_id=source_run_id, seed=args.seed,
                simulations=requested_simulations if requested_simulations is not None else 8,
            )
        else:
            if not args.run_id:
                parser.error("--numerical-benchmark requires an explicit new --run-id")
            numerical_models = ("native_c0_unchanged", "origin_eligible_annual_a0",
                                "prior_calibrated_glicko_format", "uniform")
            if args.study_model and args.study_model not in numerical_models:
                parser.error("Numerical comparisons exclude the rejected residual experiment arms")
            result = run_numerical_benchmark(
                args.run_id, source_run_id=source_run_id,
                editions=[args.edition] if args.edition else None,
                models=[args.study_model] if args.study_model else None,
                simulations=args.simulations, seed=args.seed,
                methods=tuple(args.integrations or ("monte_carlo", "hybrid")),
            )
        summary = {key: result[key] for key in (
            "run_id", "source_run_id", "simulations", "seed", "methods",
            "requested_cells", "decision", "elapsed_seconds")}
        summary["cells"] = [
            {key: row[key] for key in (
                "edition_id", "model", "method", "available", "reason",
                "requested_cell", "wall_seconds", "metrics") if key in row}
            for row in result["cells"]
        ]
        summary["comparisons"] = [
            {**{key: row[key] for key in (
                "edition_id", "model", "reference", "exact_error", "interpretation")},
             "max_absolute_differences": {
                 target: law.get("max_absolute_difference")
                 for target, law in row["targets"].items()}}
            for row in result["comparisons"]
        ]
        if args.numerical_benchmark:
            summary["report_directory"] = str(
                PROJECT_ROOT / "data/08_reporting/tournament_numerical" / args.run_id)
        print(json.dumps(_safe(summary), indent=2))
        return
    if args.source_run_id or args.integrations:
        parser.error("--source-run-id/--integrations require a numerical command")
    native_action = args.run_study or args.study_smoke or (args.native_study and (args.simulate or args.score or args.benchmark))
    if native_action and args.protocol is None:
        parser.error("Native research execution requires --protocol PATH")
    if args.freeze_study or args.prepare_native_2023:
        if not args.run_id:
            parser.error("--freeze-study/--prepare-native-2023 require an explicit new --run-id")
    if args.freeze_study:
        from src.models.tournament_improvement_runner import freeze_study
        print(freeze_study(args.run_id, research_root=args.research_root))
        return
    if args.prepare_native_2023:
        from src.models.tournament_native import prepare_2023_control
        print(json.dumps(prepare_2023_control(args.run_id, research_root=args.research_root), indent=2))
        return
    if native_action:
        from src.models.tournament_improvement_runner import (
            run_study, study_smoke, simulate_study_event, score_study_forecast, save_json, _safe,
        )
        if args.run_study or args.benchmark:
            print(json.dumps(_safe(run_study(args.protocol, research_root=args.research_root)), indent=2))
        elif args.study_smoke:
            print(json.dumps(_safe(study_smoke(args.protocol, research_root=args.research_root)), indent=2))
        elif args.simulate:
            if not args.edition or not args.study_model or not args.output:
                parser.error("Native --simulate requires --edition, --study-model and --output")
            simulate_study_event(args.protocol, args.edition, args.study_model, args.output,
                                 research_root=args.research_root, simulations=args.simulations)
            print(f"Native forecast written: {args.output}")
        elif args.score:
            if not args.forecast or not args.output:
                parser.error("Native --score requires --forecast and a new --output")
            score = score_study_forecast(args.protocol, args.forecast, research_root=args.research_root)
            save_json(args.output, score)
            print(f"Native forecast score written: {args.output}")
        return
    if args.freeze_manifest:
        freeze_manifest()
        return
    if args.smoke:
        run_smoke_test()
        return
    if args.benchmark:
        run_benchmark(run_id, simulations=args.simulations, seed=args.seed, output_base=args.output_dir)
        return
    if args.simulate:
        if args.edition not in DEFAULT_EDITIONS or not args.model:
            parser.error("Proxy --simulate requires one of the four historical --edition IDs and --model")
        if not args.output:
            parser.error("--simulate requires a new --output forecast.json")
        if args.output.exists():
            raise FileExistsError(f"Forecast already exists: {args.output}")
        res = run_simulation(args.edition, args.model, load_manifest(), TournamentPairAdapter(),
                             simulations=args.simulations, seed=args.seed)
        from src.models.tournament_improvement_runner import save_json
        save_json(args.output, res)
        print(f"Proxy forecast written: {args.output}")
        return
    if args.score:
        if not args.forecast or not args.output:
            parser.error("--score requires --forecast forecast.json and a new --output score.json")
        from src.models.tournament_improvement_runner import save_json
        forecast = json.loads(args.forecast.read_text(encoding="utf-8"))
        eid = forecast["edition_id"]
        if args.edition and args.edition != eid:
            parser.error("--edition does not match the loaded forecast")
        manifest = load_manifest()
        gt = extract_tournament_ground_truth(manifest["editions"][eid]["event_title"])
        from src.models.tournament_scorelines import ConditionalScorelineModel
        score = evaluate_tournament_edition(eid, forecast, gt, pair_table=forecast.get("pair_table"),
                                           scoreline_model=ConditionalScorelineModel())
        save_json(args.output, score)
        print(f"Proxy forecast score written: {args.output}")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
