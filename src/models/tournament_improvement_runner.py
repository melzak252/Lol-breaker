"""Bounded, source-bound retrospective native C0 tournament experiment.

No serving configuration, historical artifact, or base checkpoint is modified.
Only the authorized player filter and TRAIN conditional score law are fitted here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import re
import tempfile
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import ev_asof_replay as replay
from src.analysis.research_benchmark import losses
from src.analysis.bootstrap import monthly_block_bootstrap_delta
from src.analysis.tournament_metrics import (
    score_research_event, tournament_study_decision,
    compute_scoreline_log_loss, compute_series_length_rps,
)
from src.models.tournament_bracket_simulator import simulate_tournament_bracket
from src.models.tournament_pair_adapter import TournamentPairAdapter, normalize_team_id
from src.models.tournament_scorelines import ConditionalScorelineModel, LearnedConditionalScorelineModel
from src.models.tournament_uncertainty import PlayerResidualFilter, build_residual_tournament_arms
from src.models.tournament_formats import required_best_ofs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = PROJECT_ROOT / "data/05_model_input/tournament_improvement/tournament_c0_20261006_01/design.json"
SCHEMA = "native_c0_tournament_improvement_protocol_v1"
SEED = 20261006
ARMS = ("native_c0_unchanged", "residual_posterior_mean", "residual_independent_matched_marginals", "residual_coherent")
COMPARATORS = ("origin_eligible_annual_a0", "prior_calibrated_glicko_format", "uniform")
METRICS = ("advance_brier", "placement_rps", "champion_log_loss", "champion_brier", "finalists_brier", "swiss_record_rps", "joint_qualification_log_loss", "series_length_rps", "scoreline_log_loss")
CODE = (
    "src/models/tournament_improvement_runner.py", "src/models/tournament_native.py",
    "src/models/tournament_research_cohort.py", "src/models/tournament_uncertainty.py",
    "src/models/tournament_scorelines.py", "src/models/tournament_bracket_simulator.py",
    "src/models/tournament_formats.py", "src/models/tournament_pair_adapter.py",
    "src/models/frozen_tournament.py", "src/ratings/glicko2_core.py",
    "src/analysis/tournament_metrics.py", "src/analysis/research_benchmark.py",
    "src/analysis/bootstrap.py", "src/analysis/model_benchmark.py",
    "src/analysis/ev_asof_replay.py", "betting_app/core/models/native_c0.py",
    "betting_app/core/models/causal_a0.py", "src/models/a0_reliability.py",
    "src/models/a0_data_focus.py", "scripts/run_tournament_distribution_benchmark.py",
    "src/analysis/ev_approach_data.py", "src/models/a0_target_experiment.py",
    "src/models/a0_uncertainty.py", "src/models/a0_native_refit.py",
    "src/models/tournament_native_training.py",
)
PREFIT_RECEIPTS = frozenset({
    "design.json", "preimplementation_evidence.json",
    "union_gate_before_corrections.json", "independent_adf_smoke.json",
})


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _safe(value, location="$", nonfinite=None):
    """Strict JSON conversion with a separate reason for every nonfinite value."""
    if nonfinite is None:
        nonfinite = {}
    if isinstance(value, dict):
        return {str(k): _safe(v, f"{location}.{k}", nonfinite) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_safe(v, f"{location}[{i}]", nonfinite) for i, v in enumerate(value)]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite[location] = "positive_infinity_zero_observed_probability" if value == math.inf else "negative_infinity" if value == -math.inf else "unestimable_nan"
        return None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def save_json(path, value):
    reasons = {}
    payload = _safe(value, nonfinite=reasons)
    if reasons:
        if not isinstance(payload, dict):
            payload = {"data": payload}
        payload["nonfinite_reasons"] = reasons
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _paths(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(run_id)):
        raise ValueError("run_id must be a single safe nonempty path component")
    return {s: PROJECT_ROOT / "data" / s / "tournament_improvement" / run_id for s in ("05_model_input", "06_models", "07_model_output", "08_reporting")}


def _root(research_root):
    return replay._resolve_research_root(research_root).resolve(strict=True)


def _canonical_frame(design, root):
    reference = PROJECT_ROOT / design["base_forecasts"]
    if _sha(reference) != design["base_forecasts_sha256"]:
        raise ValueError("Canonical native C0 reference hash changed")
    frame = pd.read_parquet(reference)
    frame["golgg_match_id"] = frame.golgg_match_id.astype(str)
    if len(frame) != 11550 or frame.golgg_match_id.duplicated().any():
        raise ValueError("Canonical cohort must contain exactly 11550 unique matches")
    bank_path = root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank/metadata.parquet"
    bank = pd.read_parquet(bank_path)
    bank["golgg_match_id"] = bank.golgg_match_id.astype(str)
    if bank.golgg_match_id.duplicated().any():
        raise ValueError("Ambiguous native bank identity")
    bank = bank.set_index("golgg_match_id").loc[frame.golgg_match_id].reset_index()
    for key in ("team1_id", "team2_id", "date", "best_of", "y"):
        left = pd.to_datetime(frame[key]).dt.strftime("%Y-%m-%d") if key == "date" else frame[key].astype(str)
        right = pd.to_datetime(bank[key]).dt.strftime("%Y-%m-%d") if key == "date" else bank[key].astype(str)
        if left.tolist() != right.tolist():
            raise ValueError(f"Canonical/native bank identity mismatch: {key}")
    for key in ("roster_a", "roster_b", "result_day"):
        frame[key] = bank[key].to_list()
    label_path = root / "nonlinear-history-20260912/data/03_primary/aligned_scoreline_labels.parquet"
    labels = pd.read_parquet(label_path)
    labels["golgg_match_id"] = labels.golgg_match_id.astype(str)
    if labels.golgg_match_id.duplicated().any():
        raise ValueError("Ambiguous stopped-score labels")
    labels = labels.set_index("golgg_match_id").loc[frame.golgg_match_id]
    for key in ("best_of", "y"):
        if labels[key].astype(str).tolist() != frame[key].astype(str).tolist():
            raise ValueError(f"Stopped-score side/label mismatch: {key}")
    frame["wins_a"], frame["wins_b"] = labels.wins_a.to_numpy(int), labels.wins_b.to_numpy(int)
    for key in ("date", "result_day", "feature_source_max_day"):
        if key not in frame:
            raise ValueError(f"Canonical native provenance lacks {key}")
        frame[key] = pd.to_datetime(frame[key], errors="raise").dt.strftime("%Y-%m-%d")
    if not (frame.feature_source_max_day < frame.date).all() or not (frame.result_day >= frame.date).all():
        raise ValueError("Native feature/label chronology failed")
    frame["model_year"] = frame.date.str[:4].astype(int)
    if frame.groupby("model_year").size().to_dict() != {2024: 4642, 2025: 3805, 2026: 3103}:
        raise ValueError("Canonical annual membership changed")
    terminal = (frame.best_of.to_numpy(int) + 1) // 2
    if not np.array_equal(np.maximum(frame.wins_a, frame.wins_b), terminal) or np.any(np.minimum(frame.wins_a, frame.wins_b) >= terminal):
        raise ValueError("Illegal canonical stopped scores")
    for row in frame.itertuples(index=False):
        a, b = list(map(str, row.roster_a)), list(map(str, row.roster_b))
        if len(a) != 5 or len(b) != 5 or len(set(a)) != 5 or len(set(b)) != 5 or set(a) & set(b):
            raise ValueError(f"Invalid canonical roster {row.golgg_match_id}")
    frame.roster_a = frame.roster_a.map(lambda v: list(map(str, v)))
    frame.roster_b = frame.roster_b.map(lambda v: list(map(str, v)))
    p = frame.p_c0.to_numpy(float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid native reference probabilities")
    return frame


def freeze_study(run_id, research_root=None):
    """Freeze inputs and code, preserving and binding owned prefit evidence."""
    from src.models.tournament_native import ResearchNativePairProvider, NativeArtifactUnavailable
    from src.models.tournament_research_cohort import prepare_research_cohort
    paths, root = _paths(run_id), _root(research_root)
    input_dir = paths["05_model_input"]
    if any(paths[s].exists() for s in paths if s != "05_model_input"):
        raise FileExistsError("Study stage already exists; choose a new run_id")
    # Only these prefit receipts may predate the first freeze. Formal inputs,
    # cohort directories and any execution stage still make this namespace occupied.
    if input_dir.exists() and any(
        not p.is_file() or p.name not in PREFIT_RECEIPTS for p in input_dir.iterdir()
    ):
        raise FileExistsError(f"Study input namespace is already occupied: {input_dir}")
    design = _read_json(DESIGN_PATH)
    frame = _canonical_frame(design, root)
    input_dir.mkdir(parents=True, exist_ok=True)
    if not (input_dir / "design.json").exists():
        save_json(input_dir / "design.json", design)
    elif _read_json(input_dir / "design.json") != design:
        raise ValueError("Existing study design differs")
    preimplementation = input_dir / "preimplementation_evidence.json"
    if preimplementation.is_file():
        for relative, digest in _read_json(preimplementation)["protected_sha256"].items():
            if _sha(PROJECT_ROOT / relative) != digest:
                raise ValueError(f"Protected preimplementation artifact changed: {relative}")
    controls = {}
    prepared = PROJECT_ROOT / "data/06_models/tournament_native" / run_id / "2023"
    if prepared.is_dir():
        controls[2023] = str(prepared)
    artifacts = {}
    with ResearchNativePairProvider(research_root=root, control_roots=controls) as provider:
        for year in (2023, 2024, 2025, 2026):
            for kind in ("c0", "a0"):
                try:
                    artifacts[f"{year}:{kind}"] = provider.artifact_contract(year, model_kind=kind)
                except (NativeArtifactUnavailable, FileNotFoundError) as exc:
                    artifacts[f"{year}:{kind}"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    cohort = prepare_research_cohort(input_dir / "cohort", research_root=root,
        reference_path=PROJECT_ROOT / design["base_forecasts"], artifact_contracts=artifacts)
    memberships = {name: frame.loc[frame.model_year.eq(year), "golgg_match_id"].tolist() for name, year in (("TRAIN", 2024), ("CAL", 2025), ("ASSESS", 2026))}
    save_json(input_dir / "memberships.json", memberships)
    inventory = {}
    def bind(path):
        path = Path(path).resolve()
        if path.is_file():
            inventory[str(path)] = _sha(path)
    if preimplementation.is_file():
        for relative in _read_json(preimplementation)["protected_sha256"]:
            bind(PROJECT_ROOT / relative)
    for relative in CODE:
        bind(PROJECT_ROOT / relative)
        if not (PROJECT_ROOT / relative).is_file():
            raise FileNotFoundError(f"Study implementation missing: {relative}")
    for path in input_dir.rglob("*"):
        bind(path)
    bind(PROJECT_ROOT / design["base_forecasts"])
    bind(PROJECT_ROOT / "conf/base/research_benchmark.json")
    bind(PROJECT_ROOT / "data/artifacts/golgg-database-recovery-20260909/matches.json")
    bind(root / "nonlinear-history-20260912/data/03_primary/aligned_scoreline_labels.parquet")
    for path in (root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank").rglob("*"):
        bind(path)
    for tree in (root / replay.CAUSAL_ARCHIVE_RELATIVE, root / "nonlinear-history-20260912/code", root / "a0-phase-walkforward-20260914/code"):
        for path in tree.rglob("*.py"):
            bind(path)
    def bind_contract(value):
        if isinstance(value, dict):
            for key, item in value.items():
                bind_contract(str(key))
                bind_contract(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                bind_contract(item)
        elif isinstance(value, str):
            if "\n" in value or "://" in value or "/" not in value:
                return
            if any(len(part.encode("utf-8")) > 255 for part in Path(value).parts):
                return
            path = root / value.removeprefix("research:") if value.startswith("research:") else PROJECT_ROOT / value.removeprefix("project:") if value.startswith("project:") else Path(value)
            if path.is_absolute():
                bind(path)
            elif (PROJECT_ROOT / path).is_file():
                bind(PROJECT_ROOT / path)
            elif (root / path).is_file():
                bind(root / path)
    bind_contract(artifacts)
    bind_contract(cohort)
    protocol = {"schema": SCHEMA, "run_id": run_id, "frozen_at_utc": _now(), "design": design,
                "research_root": str(root), "paths": {s: str(p) for s, p in paths.items()},
                "control_roots": controls, "artifact_contracts": artifacts, "bindings": inventory,
                "cohort_path": str(input_dir / "cohort/manifest.json"), "membership_path": str(input_dir / "memberships.json"),
                "models": list(ARMS + COMPARATORS), "metrics": list(METRICS), "simulations": 10000,
                "seed": SEED, "bootstraps": 5000, "minimum_assessment_editions": 8,
                "base_fit_policy": "reuse_all_original_weights; explicit_separate_2023_control_only",
                "scenario": "development_exposed_historical_not_publication_certified"}
    protocol_path = input_dir / "protocol.json"
    save_json(protocol_path, protocol)
    return protocol_path


def _verify(protocol_path, research_root=None):
    protocol = _read_json(protocol_path)
    if protocol.get("schema") != SCHEMA:
        raise ValueError("Unsupported native study protocol")
    root = _root(research_root or protocol["research_root"])
    for path, digest in protocol["bindings"].items():
        actual_path = Path(path)
        old_root = Path(protocol["research_root"])
        if actual_path.is_relative_to(old_root):
            actual_path = root / actual_path.relative_to(old_root)
        if not actual_path.is_file() or _sha(actual_path) != digest:
            raise ValueError(f"Frozen study binding changed: {actual_path}")
    return protocol, root


def _new_filter(q):
    return PlayerResidualFilter(process_variance_per_day=q, prior_sd=.35, quadrature_nodes=32)


def _online(frame, state, checkpoints=None):
    """Whole decision-day prediction before queuing any label of that day."""
    predictions = pd.Series(index=frame.index, dtype=float)
    for (year, day), part in frame.sort_values(["date", "golgg_match_id"]).groupby(["model_year", "date"], sort=True):
        state.advance(day, model_year=int(year))
        for row in part.itertuples():
            predictions.at[row.Index] = state.predict(row.p_c0, row.roster_a, row.roster_b, day, model_year=int(year))["integrated_probability"]
        for row in part.itertuples():
            state.observe(row.p_c0, int(row.y), row.roster_a, row.roster_b, day, row.result_day,
                          match_id=row.golgg_match_id, model_year=int(year))
        if checkpoints is not None:
            checkpoints[day] = state.to_dict()
    return predictions


def _fit_extensions(frame, model_dir, smoke=False):
    train = frame.loc[frame.model_year.eq(2024) & frame.result_day.lt("2025-01-01")]
    cal = frame.loc[frame.model_year.eq(2025) & frame.result_day.lt("2026-01-01")]
    if len(train) != 4642 or len(cal) != 3805:
        raise ValueError("Closed TRAIN/CAL release membership changed")
    if smoke:
        chosen = train.assign(lost=np.minimum(train.wins_a, train.wins_b)).groupby(["best_of", "lost"], group_keys=False).head(24)
        train = chosen.sort_values(["date", "golgg_match_id"])
        cal = cal.sort_values(["date", "golgg_match_id"]).iloc[:60]
    cases = []
    cal_predictions = []
    for q in (0., .0001, .001):
        started = time.perf_counter()
        state = _new_filter(q)
        _online(train, state)
        save_json(model_dir / f"q_{q:g}_train_state.json", state.to_dict())
        cal_p = _online(cal, state)
        ll = float(losses(cal.y, cal_p).mean())
        cases.append({"process_variance_per_day": q, "cal_global_log_loss": ll, "train_rows": len(train), "cal_rows": len(cal), "elapsed_seconds": time.perf_counter() - started})
        cal_predictions.append(cal[["golgg_match_id", "date", "result_day", "y", "p_c0"]].assign(p_residual=cal_p, process_variance_per_day=q))
        save_json(model_dir / f"q_{q:g}_cal_state.json", state.to_dict())
    if not all(math.isfinite(c["cal_global_log_loss"]) for c in cases):
        raise ValueError("Nonfinite CAL selector loss; no assessment fallback")
    selected = min(cases, key=lambda c: (c["cal_global_log_loss"], c["process_variance_per_day"]))
    pd.concat(cal_predictions, ignore_index=True).to_parquet(model_dir / "cal_selector_predictions.parquet", index=False)
    selector = {"cases": cases, "selected": selected, "selector": "single_global_CAL2025_LL; ties_smaller_q", "assessment_used": False, "smoke": smoke}
    save_json(model_dir / "cal_selector.json", selector)
    scorelaw = LearnedConditionalScorelineModel.fit(train, probability_column="p_c0", cutoff="2025-01-01")
    save_json(model_dir / "conditional_scorelaw.json", scorelaw.to_dict())
    save_json(model_dir / "conditional_scorelaw_fit_receipt.json", {"train_rows": len(train),
              "train_max_label_day": train.result_day.max(), "fit_cutoff_exclusive": "2025-01-01",
              "earliest_eligible_year": 2025, "fit_count": 1, "assessment_labels_used": False,
              "engine_approximation": scorelaw.engine_approximation, "smoke": smoke})
    return selected["process_variance_per_day"], scorelaw, selector


def _state_at(frame, cutoff, q):
    day = pd.Timestamp(cutoff).strftime("%Y-%m-%d")
    year = int(day[:4])
    state = _new_filter(q)
    _online(frame.loc[frame.model_year.eq(year) & frame.date.lt(day)], state)
    state.advance(day, model_year=year)
    return state


def _canonical_candidate(frame, q, paths, artifact_contracts):
    assessment = frame.loc[frame.model_year.eq(2026)]
    state, checkpoints = _new_filter(q), {}
    p = _online(assessment, state, checkpoints)
    candidate = frame.copy()
    candidate["p_residual"] = candidate.p_c0
    candidate.loc[assessment.index, "p_residual"] = p
    candidate["new_forecast_row"] = candidate.model_year.eq(2026)
    candidate["update_policy"] = "released_daily_frozen_map_periods"
    for field in ("train_end", "calibration_end"):
        native_days = {year: pd.Timestamp(artifact_contracts[f"{year}:c0"][field]).strftime("%Y-%m-%d") for year in (2024, 2025, 2026)}
        candidate[f"base_{field}"] = candidate.model_year.map(native_days)
        candidate[field] = candidate[f"base_{field}"]
    candidate.loc[assessment.index, "train_end"] = max(
        candidate.loc[assessment.index, "base_train_end"].max(),
        frame.loc[frame.model_year.eq(2024), "result_day"].max())
    candidate.loc[assessment.index, "calibration_end"] = max(
        candidate.loc[assessment.index, "base_calibration_end"].max(),
        frame.loc[frame.model_year.eq(2025), "result_day"].max())
    candidate["train_end_exclusive"] = candidate.model_year.map({2024: "2023-07-01", 2025: "2024-07-01", 2026: "2025-07-01"})
    candidate["calibration_end_exclusive"] = candidate.model_year.map({2024: "2024-01-01", 2025: "2025-01-01", 2026: "2026-01-01"})
    candidate["generated_at_utc"] = _now()
    candidate["origin"] = candidate.date.str[:7]
    candidate["forecast_model"] = "player_residual_native_c0_GH32"
    if not np.array_equal(candidate.loc[~candidate.new_forecast_row, "p_residual"], frame.loc[frame.model_year.ne(2026), "p_c0"]):
        raise ValueError("Warmup probabilities changed")
    candidate.to_parquet(paths["07_model_output"] / "candidate.parquet", index=False)
    save_json(paths["06_models"] / "assessment_final_state.json", state.to_dict())
    with (paths["06_models"] / "assessment_daily_states.jsonl").open("x") as stream:
        for day, payload in checkpoints.items():
            stream.write(json.dumps({"day": day, "state": payload}, allow_nan=False) + "\n")
    delta = losses(assessment.y, p) - losses(assessment.y, assessment.p_c0)
    guard_estimate = _paired_unclipped(assessment, delta)
    brier = float(np.mean((assessment.y.to_numpy() - p.to_numpy()) ** 2 - (assessment.y.to_numpy() - assessment.p_c0.to_numpy()) ** 2))
    slices = []
    for column in ("best_of", "competition_tier", "tier", "competition_region", "unknown_players", "history_stat_coverage"):
        if column not in assessment:
            continue
        for value, part in assessment.groupby(column, dropna=False):
            sub_p = p.loc[part.index]
            d = losses(part.y, sub_p) - losses(part.y, part.p_c0)
            slices.append({"slice": column, "value": str(value), "log_loss": _paired_unclipped(part, d),
                           "brier_delta": float(np.mean((part.y - sub_p) ** 2 - (part.y - part.p_c0) ** 2))})
    guard = {"log_loss_delta_ci_upper": guard_estimate["ci95"][1] if guard_estimate["ci95"] else None,
             "brier_delta": brier, "coverage_complete": len(candidate) == 11550,
             "structural_checks_passed": bool(np.isfinite(candidate.p_residual).all() and candidate.p_residual.between(0, 1).all()),
             "chronology_checks_passed": bool((candidate.train_end < candidate.date).all() and (candidate.calibration_end < candidate.date).all()),
             "monthly_log_loss": guard_estimate, "slices": slices, "rows": 3103, "warmup_rows": 8447}
    monthly = []
    for month, part in assessment.groupby(assessment.date.str[:7], sort=True):
        sub_p = p.loc[part.index]
        monthly.append({"month": month, "rows": len(part), "native_c0_log_loss": float(losses(part.y, part.p_c0).mean()),
                        "candidate_log_loss": float(losses(part.y, sub_p).mean()),
                        "native_c0_brier": float(np.mean((part.y - part.p_c0) ** 2)),
                        "candidate_brier": float(np.mean((part.y - sub_p) ** 2))})
    guard["monthly_metrics"] = monthly
    save_json(paths["08_reporting"] / "series_guard.json", guard)
    return candidate, guard


def _paired_unclipped(frame, delta):
    if not np.isfinite(delta).all():
        return {"delta": None, "ci95": None, "rows": len(frame), "blocks": int(frame.date.str[:7].nunique()),
                "unavailable_reason": "Nonfinite unclipped paired loss; observed zero probability or undefined infinity contrast"}
    months = frame.date.str[:7]
    result = {"delta": float(np.mean(delta)), "rows": len(frame), "blocks": int(months.nunique()),
              "equal_block_delta": float(pd.Series(np.asarray(delta)).groupby(months.to_numpy()).mean().mean()),
              "ci95": None, "seed": SEED, "bootstraps": 5000}
    if result["blocks"] > 1:
        work = pd.DataFrame({"date": frame.date.to_numpy(), "delta": np.asarray(delta)})
        mean, lo, hi, samples = monthly_block_bootstrap_delta(work, "delta", n_bootstraps=5000, random_seed=SEED)
        result.update(delta=mean, ci95=[lo, hi], bootstrap_p_nonnegative=float(np.mean(samples >= 0)))
    return result


def _scorelaw_diagnostics(candidate, law, report):
    """Continuous hypothetical-series diagnostics, not observed-reach conditioning."""
    rows = []
    fixed = ConditionalScorelineModel(beta_momentum=.35)
    for row in candidate.loc[candidate.model_year.eq(2026)].itertuples(index=False):
        for model, p in (("native_c0_unchanged", row.p_c0), ("residual_GH32_integrated", row.p_residual)):
            for law_name in ("learned_continuous_TRAIN", "pooled_TRAIN", "fixed_markov_0.35"):
                if law_name == "pooled_TRAIN":
                    terminal = (int(row.best_of) + 1) // 2
                    losing = [1.] if row.best_of == 1 else law.pooled_distributions[int(row.best_of)]
                    pmf = {(terminal, k): p * losing[k] for k in range(terminal)}
                    pmf.update({(k, terminal): (1 - p) * losing[k] for k in range(terminal)})
                    length = {terminal + k: losing[k] for k in range(terminal)}
                else:
                    active = law if law_name == "learned_continuous_TRAIN" else fixed
                    pmf = active.scoreline_pmf(float(p), int(row.best_of))
                    length = active.series_length_pmf(float(p), int(row.best_of))
                rows.append({"golgg_match_id": row.golgg_match_id, "date": row.date, "model": model, "score_law": law_name,
                             "scoreline_log_loss": compute_scoreline_log_loss(pmf, (int(row.wins_a), int(row.wins_b))),
                             "series_length_rps": compute_series_length_rps(length, int(row.wins_a + row.wins_b), int(row.best_of))})
    save_json(report / "series_scorelaw_diagnostics.json", {"scope": "Frozen-origin hypothetical series; continuous law at declared series probability, not path/reach conditioned; coherent bracket finite-draw compiled laws scored separately", "rows": rows})
    pd.DataFrame(_safe(rows)).to_csv(report / "series_scorelaw_diagnostics.csv", index=False)


def _pair_mapping(table):
    result = {}
    for row in table["rows"]:
        key = (str(row.get("A", row.get("team_a"))), str(row.get("B", row.get("team_b"))), int(row.get("BO", row.get("best_of"))))
        if key in result:
            raise ValueError(f"Duplicate pair row: {key}")
        result[key] = float(row["p"])
    return result


def _glicko_table(event):
    """Reuse the existing strictly prior-CAL format Glicko recipe, with exact IDs."""
    adapter = TournamentPairAdapter()
    names = event["source_provenance"]["team_source_names"]
    source_ids = {t: normalize_team_id(names[t]) for t in event["teams"]}
    if len(set(source_ids.values())) != len(source_ids):
        raise ValueError("Archived source-name map does not uniquely resolve native entrants")
    formats = sorted(required_best_ofs(event["spec"]))
    table = adapter.predict_pair_table(list(source_ids.values()), event["cutoff"], "glicko_format", best_ofs=formats)
    receipt = adapter.predictor_receipt(list(source_ids.values()), event["cutoff"], "glicko_format", best_ofs=formats)
    rows = [{"A": a, "B": b, "BO": bo, "p": table[source_ids[a], source_ids[b], bo]}
            for a, b in combinations(event["teams"], 2) for bo in formats]
    return {"rows": rows, "probability_unit": "series_win", "provenance": receipt,
            "native_to_source_ids": source_ids}




def _simulate_cell(event, model, table, law, draws, output_dir, simulations=10000, residual_metadata=None, filter_state=None):
    forecast = simulate_tournament_bracket(event["spec"], _pair_mapping(table),
        conditional_score_distributions=law.conditional_distributions(), simulations=simulations, seed=SEED,
        team_strength_draws=draws)
    payload = {"schema": "native_research_tournament_forecast_v1", "event_id": event["id"], "edition_id": event["edition_id"],
               "model": model, "event": event, "pair_table": table, "scoreline_model": law.to_dict() if hasattr(law, "to_dict") else {"kind": "fixed_markov", "beta_momentum": .35}, "forecast": forecast}
    payload["team_strength_draws"] = draws
    payload["residual_metadata"] = residual_metadata
    payload["filter_state"] = filter_state
    payload["series_score_scope"] = "frozen_origin_unconditional_hypothetical_pair_not_reach_conditioned"
    save_json(output_dir / f"{event['id']}__{model}.json", payload)
    return forecast


def _event_cases(event, frame, q, provider, models=ARMS + COMPARATORS, nonformal_smoke=False, simulations=10000, native_tables=None):
    from src.models.tournament_native import NativeArtifactUnavailable
    from src.models.tournament_research_cohort import validate_research_event
    if not nonformal_smoke:
        validate_research_event(event)
    cases, unavailable = {}, {}
    base = None
    def native_table(kind):
        if native_tables is None:
            return provider.predict_table(event, model_kind=kind)
        value = native_tables[event["id"], kind]
        if isinstance(value, Exception):
            raise value
        return value
    try:
        base = native_table("c0")
        cases["native_c0_unchanged"] = {"table": base, "team_strength_draws": None}
    except (NativeArtifactUnavailable, FileNotFoundError) as exc:
        for model in ARMS:
            unavailable[model] = f"{type(exc).__name__}: {exc}"
    for model in ARMS[1:]:
        if int(event["year"]) < 2026:
            unavailable[model] = "Extension selected on CAL2025; unavailable before 2026 origin"
    if base is not None and int(event["year"]) >= 2026 and any(m in models for m in ARMS[1:]):
        state = _state_at(frame, event["cutoff"], q)
        cases.update(build_residual_tournament_arms(base, event, state, simulations=simulations, seed=SEED))
        snapshot = state.to_dict()
        for model in ARMS:
            cases[model]["filter_state"] = snapshot
    if "origin_eligible_annual_a0" in models:
        try:
            cases["origin_eligible_annual_a0"] = {"table": native_table("a0"), "team_strength_draws": None}
        except (NativeArtifactUnavailable, FileNotFoundError) as exc:
            unavailable["origin_eligible_annual_a0"] = f"{type(exc).__name__}: {exc}"
    if "prior_calibrated_glicko_format" in models:
        try:
            cases["prior_calibrated_glicko_format"] = {"table": _glicko_table(event), "team_strength_draws": None}
        except (KeyError, ValueError) as exc:
            unavailable["prior_calibrated_glicko_format"] = f"Prior-source Glicko unavailable: {exc}"
    if "uniform" in models:
        cases["uniform"] = {"table": {"rows": [{"A": a, "B": b, "BO": bo, "p": .5} for a, b in combinations(event["teams"], 2) for bo in required_best_ofs(event["spec"])], "probability_unit": "series_win", "provenance": {"recipe": "uniform_reference"}}, "team_strength_draws": None}
    return {m: cases[m] for m in models if m in cases and m not in unavailable}, unavailable


def _execute(protocol_path, research_root=None, smoke=False, paths=None):
    from src.models.tournament_native import ResearchNativePairProvider
    started = time.perf_counter()
    protocol, root = _verify(protocol_path, research_root)
    if paths is None:
        paths = {s: Path(p) for s, p in protocol["paths"].items()}
    if any(paths[s].exists() for s in ("06_models", "07_model_output", "08_reporting")):
        raise FileExistsError("A study execution stage already exists; no resume or overwrite")
    for stage in ("06_models", "07_model_output", "08_reporting"):
        paths[stage].mkdir(parents=True, exist_ok=False)
    report = paths["08_reporting"]
    save_json(report / "execution.json", {"status": "STARTED", "protocol_sha256": _sha(protocol_path), "started_at_utc": _now(), "smoke": smoke, "base_fits": 0, "python": platform.python_version()})
    try:
        frame = _canonical_frame(protocol["design"], root)
        q, learned, selector = _fit_extensions(frame, paths["06_models"], smoke)
        if smoke:
            guard = {"log_loss_delta_ci_upper": None, "brier_delta": None, "coverage_complete": False, "structural_checks_passed": True, "chronology_checks_passed": True, "reason": "Nonformal smoke; no canonical assessment/adoption"}
        else:
            candidate, guard = _canonical_candidate(frame, q, paths, protocol["artifact_contracts"])
            _scorelaw_diagnostics(candidate, learned, report)
        cohort = _read_json(protocol["cohort_path"])
        events = list(cohort["events"].values())
        if smoke:
            events = [next(e for e in events if int(e["year"]) == 2026 and e.get("spec"))]
        scores, coverage = [], []
        with ResearchNativePairProvider(research_root=root, control_roots=protocol["control_roots"]) as provider:
            native_tables = None
            if not smoke:
                from src.models.tournament_native import NativeArtifactUnavailable
                from src.models.tournament_research_cohort import validate_research_event
                native_tables = {}
                ordered_events = sorted((e for e in events if e.get("spec")), key=lambda e: e["cutoff"])
                for event in ordered_events:
                    validate_research_event(event)
                # Advance one genuine history chronologically per backend, rather
                # than rebuilding the same fourteen-year prefix for every event.
                for kind in ("c0", "a0"):
                    for event in ordered_events:
                        try:
                            native_tables[event["id"], kind] = provider.predict_table(event, model_kind=kind)
                        except (NativeArtifactUnavailable, FileNotFoundError) as exc:
                            native_tables[event["id"], kind] = exc
                provider.close()
            for event in events:
                event_started = time.perf_counter()
                if smoke:
                    from src.models.tournament_research_cohort import validate_research_event
                    validate_research_event(event)
                    event = copy.deepcopy(event)
                    observed = _event_labels(event)
                    if observed is not None:
                        opening_label = next(m for m in observed["matches"] if int(m["best_of"]) == 5)
                        event["teams"] = [opening_label["team_a"], opening_label["team_b"]]
                    else:
                        event["teams"] = event["teams"][:2]
                    event["rosters"] = {t: event["rosters"][t] for t in event["teams"]}
                    event["spec"] = {"version": 1, "id": "native_two_entrant_nonformal_smoke", "teams": event["teams"], "stages": [{"id": "final", "kind": "single_elimination", "entrants": event["teams"], "best_of": 5}], "champion": {"stage": "final", "group": "all", "rank": 1}}
                    event["source_provenance"]["smoke_subscenario"] = "two_real_entrants_not_historical_tournament_evidence"
                if not event.get("spec"):
                    for model in ARMS + COMPARATORS:
                        reason = event.get("model_eligibility", {}).get(model, {}).get("reason") or "No source-bound supported event specification"
                        coverage.append({"event_id": event["id"], "edition_id": event["edition_id"], "model": model, "available": False, "reason": reason})
                        scores.append(_unavailable_row(event, model, reason))
                    continue
                cases, missing = _event_cases(event, frame, q, provider, models=ARMS if smoke else ARMS + COMPARATORS, nonformal_smoke=smoke, simulations=100 if smoke else 10000, native_tables=native_tables)
                labels = _event_labels(event)
                if smoke and labels is not None:
                    # The reduced bracket has no historical champion label; score real series only.
                    labels = {"event_id": event["id"], "edition_id": event["edition_id"], "matches": [m for m in labels.get("matches", []) if {m["team_a"], m["team_b"]}.issubset(set(event["teams"]))], "champion": None, "finalists": None, "qualifiers": {}, "placement_bands": None, "swiss_records": None}
                for model in (ARMS if smoke else ARMS + COMPARATORS):
                    if model not in cases:
                        coverage.append({"event_id": event["id"], "edition_id": event["edition_id"], "model": model, "available": False, "reason": missing[model]})
                        scores.append(_unavailable_row(event, model, missing[model]))
                        continue
                    cell = cases[model]
                    law = learned if int(event["year"]) >= 2025 else ConditionalScorelineModel(beta_momentum=.35)
                    forecast = _simulate_cell(event, model, cell["table"], law, cell["team_strength_draws"], paths["07_model_output"], simulations=100 if smoke else 10000, residual_metadata=cell.get("residual_metadata"), filter_state=cell.get("filter_state"))
                    scored = _score_event(event, forecast, labels, cell["table"], law)
                    row = {"event_id": event["id"], "edition_id": event["edition_id"], "year": event["year"], "model": model, "eligible": True,
                           "metrics": {k: scored[k] for k in METRICS}, "target_counts": scored["target_counts"],
                           "structural_checks_passed": True, "chronology_checks_passed": True, "unavailable_reasons": scored["unavailable_reasons"], "roster_basis": event["roster_basis"]}
                    scores.append(row)
                    coverage.append({"event_id": event["id"], "edition_id": event["edition_id"], "model": model, "available": True, "reason": None, "forecast_path": str(paths["07_model_output"] / f"{event['id']}__{model}.json"), "seconds_since_event_start": time.perf_counter() - event_started})
        ledger_complete = len(coverage) == len(events) * (len(ARMS) if smoke else len(ARMS + COMPARATORS))
        guard["coverage_complete"] = bool(guard["coverage_complete"] and ledger_complete)
        decision = tournament_study_decision(scores, guard, minimum_editions=8, bootstraps=5000, seed=SEED)
        if smoke:
            decision.update(decision="NONFORMAL_SMOKE_ONLY", adoption_allowed=False)
        save_json(report / "metrics.json", {"scores": scores, "metrics": list(METRICS)})
        csv_rows = [{**{k: r[k] for k in ("event_id", "edition_id", "year", "model", "eligible", "roster_basis")}, **_safe(r["metrics"]),
                     "target_counts": json.dumps(r["target_counts"], sort_keys=True),
                     "unavailable_reasons": json.dumps(r["unavailable_reasons"], sort_keys=True),
                     "nonfinite_reasons": json.dumps({k: "zero_observed_probability_infinite_unclipped_loss" for k, v in r["metrics"].items() if isinstance(v, (float, np.floating)) and not np.isfinite(v)}, sort_keys=True)} for r in scores]
        pd.DataFrame(csv_rows).to_csv(report / "metrics.csv", index=False)
        save_json(report / "coverage.json", {"cells": coverage, "exclusions": cohort["exclusions"], "requested_models": list(ARMS + COMPARATORS), "complete_ledger": ledger_complete})
        save_json(report / "sources.json", {"source_sha256": cohort["source_sha256"], "bindings": protocol["bindings"], "artifact_contracts": protocol["artifact_contracts"]})
        save_json(report / "decision.json", decision)
        # Recheck all frozen inputs/code/assets after inference; this is not a repair path.
        _verify(protocol_path, root)
        output_hashes = {str(path): _sha(path) for stage in ("06_models", "07_model_output", "08_reporting")
                         for path in paths[stage].rglob("*") if path.is_file()}
        receipt = {"status": "COMPLETED", "decision": decision["decision"], "run_id": protocol["run_id"], "smoke": smoke,
                   "protocol_sha256": _sha(protocol_path), "selector": selector, "base_fits": 0,
                   "filter_train_passes": 3, "scorelaw_train_fits": 1, "assessment_refits": 0,
                   "simulations_per_case": 100 if smoke else 10000, "seed": SEED,
                   "completed_at_utc": _now(), "elapsed_seconds": time.perf_counter() - started,
                   "forecast_cells": len(scores), "outputs": {s: str(p) for s, p in paths.items()},
                   "source_nonmutation": True, "publication_certification": False}
        receipt["output_sha256"] = output_hashes
        receipt["forecast_cells"] = sum(bool(row["available"]) for row in coverage)
        receipt["eligibility_cells"] = len(coverage)
        save_json(report / "receipt.json", receipt)
        save_json(report / "formal_execution.json", {**receipt, "formal": not smoke})
        return receipt if smoke else decision
    except Exception as exc:
        save_json(report / "failure.json", {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": time.perf_counter() - started, "smoke": smoke})
        raise


def _unavailable_row(event, model, reason):
    return {"event_id": event["id"], "edition_id": event["edition_id"], "year": event["year"], "model": model, "eligible": False,
            "metrics": {k: None for k in METRICS}, "target_counts": {k: 0 for k in METRICS},
            "structural_checks_passed": False, "chronology_checks_passed": False,
            "unavailable_reasons": {k: reason for k in METRICS}, "roster_basis": event.get("roster_basis")}

def _event_labels(event):
    if event.get("labels_available") is False:
        return None
    return _read_json(event["labels_path"])


def _score_event(event, forecast, labels, pair_table, law):
    if labels is None:
        reason = event["labels_unavailable_reason"]
        return {"event_id": event["id"], "edition_id": event["edition_id"],
                **{metric: None for metric in METRICS},
                "target_counts": {metric: 0 for metric in METRICS},
                "unavailable_reasons": {metric: reason for metric in METRICS}}
    return score_research_event(event, forecast, labels, pair_table=pair_table, scoreline_model=law)


def run_study(protocol_path, research_root=None, smoke=False):
    if smoke:
        return study_smoke(protocol_path, research_root=research_root)
    return _execute(protocol_path, research_root=research_root)


def study_smoke(protocol_path, research_root=None):
    """Run real filters/law/cohort/native tensors and a minimal real-entrant bracket."""
    parent = PROJECT_ROOT / "data/08_reporting/tournament_improvement_smoke"
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="nonformal_", dir=parent))
    paths = {s: directory / s for s in ("05_model_input", "06_models", "07_model_output", "08_reporting")}
    return _execute(protocol_path, research_root=research_root, smoke=True, paths=paths)


def simulate_study_event(protocol_path, edition_id, model, output_path, research_root=None, simulations=10000):
    """Use frozen study policy and saved selector, never fit on a CLI simulation."""
    from src.models.tournament_native import ResearchNativePairProvider
    protocol, root = _verify(protocol_path, research_root)
    if model not in ARMS + COMPARATORS:
        raise ValueError(f"Native study model must be one of {ARMS + COMPARATORS}")
    events = _read_json(protocol["cohort_path"])["events"]
    event = next((e for e in events.values() if e["id"] == edition_id or e["edition_id"] == edition_id), None)
    if event is None:
        raise ValueError(f"No frozen research event: {edition_id}")
    model_dir = Path(protocol["paths"]["06_models"])
    q = _read_json(model_dir / "cal_selector.json")["selected"]["process_variance_per_day"]
    learned = LearnedConditionalScorelineModel.from_dict(_read_json(model_dir / "conditional_scorelaw.json"))
    frame = _canonical_frame(protocol["design"], root)
    with ResearchNativePairProvider(research_root=root, control_roots=protocol["control_roots"]) as provider:
        cases, missing = _event_cases(event, frame, q, provider, models=(model,), simulations=simulations)
        if model not in cases:
            raise ValueError(missing[model])
        cell = cases[model]
        law = learned if int(event["year"]) >= 2025 else ConditionalScorelineModel(beta_momentum=.35)
        forecast = simulate_tournament_bracket(event["spec"], _pair_mapping(cell["table"]), conditional_score_distributions=law.conditional_distributions(), simulations=simulations, seed=SEED, team_strength_draws=cell["team_strength_draws"])
        payload = {"schema": "native_research_tournament_forecast_v1", "event": event, "event_id": event["id"], "edition_id": event["edition_id"], "model": model, "pair_table": cell["table"], "forecast": forecast, "scoreline_model": learned.to_dict() if int(event["year"]) >= 2025 else {"kind": "fixed_markov", "beta_momentum": .35}, "protocol_sha256": _sha(protocol_path)}
        payload["team_strength_draws"] = cell["team_strength_draws"]
        payload["residual_metadata"] = cell.get("residual_metadata")
        payload["filter_state"] = cell.get("filter_state")
        save_json(output_path, payload)
    return payload


def score_study_forecast(protocol_path, forecast_path, research_root=None):
    protocol, _ = _verify(protocol_path, research_root)
    payload = _read_json(forecast_path)
    event = payload["event"]
    events = _read_json(protocol["cohort_path"])["events"]
    if event not in events.values():
        raise ValueError("Forecast event differs from frozen study input")
    model_data = payload["scoreline_model"]
    law = ConditionalScorelineModel(beta_momentum=model_data["beta_momentum"]) if model_data.get("kind") == "fixed_markov" else LearnedConditionalScorelineModel.from_dict(model_data)
    return _score_event(event, payload["forecast"], _event_labels(event), payload["pair_table"], law)
