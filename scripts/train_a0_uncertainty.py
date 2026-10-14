"""Train the preregistered genuine-A0 ensembles, then use the canonical benchmark.

The protocol is recorded before fitting. Sources in the research archive are
read-only dependencies; models, predictions and reports use repository stages.
This entry point does not connect to the application database or serve models.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.analysis.research_benchmark import attach_candidate, run_suite, sha

BANK_RELATIVE = Path("a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank")
VARIANTS = ("seed_only", "block_bootstrap")
YEARS = (2024, 2025, 2026)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def train_job(research_root, model_dir, prediction_path, year, variant, seeds, device, resume=False):
    from src.models.a0_uncertainty import train_year

    return train_year(
        Path(research_root), Path(model_dir), Path(prediction_path),
        year=year, variant=variant, seeds=tuple(seeds), threads=2, device=device,
        resume=resume,
    )


def check_probabilities(frame: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """Independent arithmetic checks; the suite checks complete identity/coverage."""
    results = {}
    for variant in VARIANTS:
        part = frame.loc[(frame.variant == variant) & (frame.partition == "test")]
        attach_candidate(baseline, part, "p")
        members = part[[f"p_member_{i}" for i in range(5)]].to_numpy(float)
        mean = members.mean(axis=1)
        variance = members.var(axis=1)
        aleatoric = (members * (1.0 - members)).mean(axis=1)
        if not np.isfinite(members).all() or ((members <= 0) | (members >= 1)).any():
            raise ValueError("invalid trained member probabilities")
        checks = {
            "mean_error": np.max(np.abs(part.p.to_numpy() - mean)),
            "variance_error": np.max(np.abs(part.epistemic_variance.to_numpy() - variance)),
            "aleatoric_error": np.max(np.abs(part.aleatoric_variance.to_numpy() - aleatoric)),
            "total_variance_error": np.max(np.abs(aleatoric + variance - mean * (1.0 - mean))),
            "std_error": np.max(np.abs(part.p_std.to_numpy() - np.sqrt(variance))),
            "q10_error": np.max(np.abs(part.p_q10.to_numpy() - np.quantile(members, 0.1, axis=1))),
            "q90_error": np.max(np.abs(part.p_q90.to_numpy() - np.quantile(members, 0.9, axis=1))),
        }
        if any(not np.isfinite(value) or value > 1e-12 for value in checks.values()):
            raise ValueError(f"uncertainty arithmetic mismatch: {variant}: {checks}")
        results[variant] = {"rows": len(part), **{key: float(value) for key, value in checks.items()}}
    return results


def check_saved_inference(model_root: Path, research_root: Path, predictions: pd.DataFrame) -> dict:
    """Exercise saved models on real, swapped, missing-context and cold rosters."""
    from src.models.a0_uncertainty import load_model, summarize_members

    bank = research_root / BANK_RELATIVE
    metadata = pd.read_parquet(bank / "metadata.parquet", columns=["golgg_match_id"])
    row_for_id = pd.Series(np.arange(len(metadata)), index=metadata.golgg_match_id.astype(str))
    with np.load(bank / "arrays.npz", allow_pickle=False) as archive:
        raw_bank = {key: archive[key] for key in archive.files}
    history_bank = {
        key: np.load(bank / f"{key}.npy", mmap_mode="r")
        for key in ("history", "mask", "champions")
    }
    results = {}
    for year in YEARS:
        for variant in VARIANTS:
            part = predictions.loc[
                (predictions.model_year == year) & (predictions.variant == variant)
                & (predictions.partition == "test")
            ].groupby("best_of", sort=True).head(3)
            if set(part.best_of) != {1, 3, 5}:
                raise ValueError(f"inference smoke lacks BO coverage: {year}")
            indices = row_for_id.loc[part.golgg_match_id.astype(str)].to_numpy()
            raw = {key: value[indices] for key, value in raw_bank.items()}
            hist = {key: value[indices] for key, value in history_bank.items()}
            input_arrays = {**raw, **hist}
            before = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in input_arrays.items()}
            swapped_raw = {key: value[:, ::-1].copy() for key, value in raw.items()}
            swapped_hist = {key: value[:, ::-1].copy() for key, value in hist.items()}
            cold_raw = {key: np.zeros_like(value) for key, value in raw.items()}
            cold_hist = {key: np.zeros_like(value) for key, value in hist.items()}
            bo = part.best_of.to_numpy(int)
            model = load_model(model_root / variant / str(year), research_root)
            checks = {}
            for mode, column in (
                ("full", "p"), ("no_w20", "p_no_w20"), ("no_organization", "p_no_organization")
            ):
                members = model.predict_members(raw, hist, bo, mode=mode)
                reverse = model.predict_members(swapped_raw, swapped_hist, bo, mode=mode)
                cold = model.predict_members(cold_raw, cold_hist, bo, mode=mode)
                checks[f"{mode}_swap_error"] = float(np.max(np.abs(members + reverse - 1.0)))
                checks[f"{mode}_cold_error"] = float(np.max(np.abs(cold - 0.5)))
                checks[f"{mode}_stored_error"] = float(np.max(np.abs(members.mean(axis=1) - part[column].to_numpy())))
                if mode == "full":
                    stored = part[[f"p_member_{i}" for i in range(5)]].to_numpy()
                    checks["member_reload_error"] = float(np.max(np.abs(members - stored)))
                    cold_stats = summarize_members(cold)
                    checks["cold_total_variance_error"] = float(np.max(np.abs(cold_stats["total_variance"] - 0.25)))
                    checks["cold_epistemic_variance"] = float(np.max(cold_stats["epistemic_variance"]))
                absent = {key: value.copy() for key, value in raw.items()}
                if mode == "no_w20":
                    absent["gates"][..., 1] = 0
                elif mode == "no_organization":
                    absent["gates"][...] = 0
                else:
                    continue
                routed = model.predict_members(absent, hist, bo, mode="full")
                unavailable = {key: value.copy() for key, value in absent.items()}
                unavailable["w20"].fill(0)
                if mode == "no_organization":
                    unavailable["team"].fill(0)
                perturbed = model.predict_members(unavailable, hist, bo, mode="full")
                checks[f"{mode}_masked_values_error"] = float(np.max(np.abs(routed - perturbed)))
            after = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in input_arrays.items()}
            if before != after:
                raise ValueError("saved-model inference mutated its inputs")
            if any(not np.isfinite(value) or value > 2e-6 for value in checks.values()):
                raise ValueError(f"saved-model inference mismatch: {variant}/{year}: {checks}")
            results[f"{variant}/{year}"] = {"rows": len(part), "input_mutation": False, **checks}
    return results


def run(
    protocol_path: Path, research_root: Path, workers: int = 2,
    device: str = "cpu", resume_checkpoints: bool = False,
) -> dict:
    started = time.monotonic()
    protocol = json.loads(protocol_path.read_text())
    run_id = protocol["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("run_id must be a simple directory name")
    if protocol.get("schema_version") != 1 or protocol.get("status") != "PREREGISTERED":
        raise ValueError("a preregistered schema-version-1 protocol is required")
    training = protocol["training"]
    if tuple(protocol["fixed_variants"]) != VARIANTS or tuple(training["years"]) != YEARS:
        raise ValueError("this experiment requires both variants and the complete 2024-2026 cohort")
    if [training[key] for key in ("nn_epochs", "warmup_mlp_epochs", "full_mlp_epochs")] != [12, 24, 12]:
        raise ValueError("the genuine A0 training schedule must remain 12/24/12 epochs")
    seeds = tuple(protocol["seeds"])
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("five distinct preregistered member seeds are required")
    if workers < 1 or workers > training["max_workers"] or training["torch_threads_per_worker"] != 2:
        raise ValueError("worker/thread count exceeds the frozen resource allocation")
    if device not in ("cpu", "cuda") or (device == "cuda" and workers != 1):
        raise ValueError("CUDA training uses one worker; device must be cpu or cuda")
    draws = int(protocol["evaluation"]["bootstrap_draws"])
    if draws < 5000:
        raise ValueError("at least 5000 monthly bootstrap draws are required")

    model_root = PROJECT_ROOT / "data/06_models/a0_uncertainty" / run_id
    prediction_root = PROJECT_ROOT / "data/07_model_output/a0_uncertainty" / run_id
    report_root = PROJECT_ROOT / "data/08_reporting/a0_uncertainty" / run_id
    input_manifest_path = protocol_path.parent / "input_manifest.json"
    completed_path = report_root / "completed.json"
    if resume_checkpoints:
        if not input_manifest_path.is_file():
            raise FileNotFoundError(f"checkpoint resumption requires the original input manifest: {input_manifest_path}")
        if any(not path.is_dir() for path in (model_root, prediction_root, report_root)):
            raise FileNotFoundError("checkpoint resumption requires all partial experiment output roots")
        if completed_path.exists():
            raise FileExistsError(f"refusing to resume an already-completed experiment: {completed_path}")
        original_manifest = json.loads(input_manifest_path.read_text())
    else:
        for path in (model_root, prediction_root, report_root, input_manifest_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite an experiment artifact: {path}")
        original_manifest = None

    config_path = PROJECT_ROOT / protocol["source_manifest"]
    doctor = run_suite(config_path, research_root, doctor=True)
    if doctor["status"] != "READY":
        raise ValueError(f"canonical inputs unavailable or changed: {doctor}")
    bank = research_root / BANK_RELATIVE
    bank_receipt = json.loads((bank / "completed.json").read_text())
    if bank_receipt["status"] not in ("COMPLETE", "PASS"):
        raise ValueError("the corrected feature bank is not complete")
    for name, expected in bank_receipt["outputs_sha256"].items():
        if sha(bank / name) != expected:
            raise ValueError(f"corrected feature-bank hash mismatch: {name}")
    input_paths = [
        bank / name for name in ("completed.json", "metadata.parquet", "arrays.npz", "history.npy", "mask.npy", "champions.npy")
    ] + [
        research_root / "nonlinear-history-20260912/data/03_primary/aligned_scoreline_labels.parquet",
        research_root / "auxiliary-player-20260912/data/03_primary/targets.npy",
        research_root / "auxiliary-player-20260912/data/03_primary/target_mask.npy",
        research_root / "auxiliary-player-20260912/data/03_primary/schema.json",
        research_root / "unified-ratings-20260911/data/04_feature/roster-optional-v1/metadata.jsonl",
    ]
    code_paths = [
        PROJECT_ROOT / "src/models/a0_accelerator.py",
        Path(__file__), PROJECT_ROOT / "src/models/a0_uncertainty.py",
        PROJECT_ROOT / "src/analysis/a0_uncertainty.py", config_path,
        PROJECT_ROOT / "src/analysis/research_benchmark.py",
        PROJECT_ROOT / "src/analysis/model_benchmark.py",
        PROJECT_ROOT / "src/analysis/bootstrap.py",
        PROJECT_ROOT / "src/analysis/metrics.py",
    ]
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "joblib")}
    inputs = {str(path.relative_to(research_root)): sha(path) for path in input_paths}
    current_code = {str(path.relative_to(PROJECT_ROOT)): sha(path) for path in code_paths}
    if resume_checkpoints:
        if original_manifest["run_id"] != run_id:
            raise ValueError("run ID differs from the original input manifest")
        if str(research_root) != original_manifest["research_root"]:
            raise ValueError("research input root differs from the original input manifest")
        if sha(protocol_path) != original_manifest["protocol_sha256"]:
            raise ValueError("preregistered protocol differs from the original input manifest")
        if inputs != original_manifest["inputs"]:
            raise ValueError("source-data hashes differ from the original input manifest")
        if versions != original_manifest["versions"]:
            raise ValueError("package versions differ from the original input manifest")
        if device != original_manifest["training_device"]:
            raise ValueError("training device differs from the original input manifest")
        execution_manifest_path = report_root / f"resumption_manifest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}.json"
        if execution_manifest_path.exists():
            raise FileExistsError(f"resumption manifest already exists: {execution_manifest_path}")
        checkpoint_hashes = {
            str(path.relative_to(model_root)): sha(path)
            for path in sorted(model_root.rglob("*.joblib")) if path.is_file()
        }
    else:
        execution_manifest_path = input_manifest_path
        checkpoint_hashes = {}
    input_manifest = {
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(protocol_path),
        "research_root": str(research_root),
        "inputs": inputs,
        "code": current_code,
        "canonical_doctor": doctor,
        "versions": versions,
        "python": sys.version,
        "workers": workers, "torch_threads_per_worker": 2,
        "training_device": device,
        "seeds": list(seeds), "years": list(YEARS), "variants": list(VARIANTS),
        "execution_environment": {
            key: os.environ.get(key) for key in (
                "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }
    if resume_checkpoints:
        input_manifest.update({
            "resumption": True,
            "parent_input_manifest": {
                "path": str(input_manifest_path.relative_to(PROJECT_ROOT)),
                "sha256": sha(input_manifest_path),
            },
            "original_code": original_manifest["code"],
            "pre_resume_component_checkpoint_sha256": checkpoint_hashes,
        })
        preserved_sources = {}
        for relative, expected in original_manifest["code"].items():
            snapshot = input_manifest_path.parent / "executed_source_before_receipt_fix" / relative
            if current_code[relative] != expected and not snapshot.is_file():
                raise FileNotFoundError(f"changed execution code requires its original snapshot: {relative}")
            if snapshot.is_file():
                if sha(snapshot) != expected:
                    raise ValueError(f"preserved pre-fix source snapshot hash mismatch: {relative}")
                preserved_sources[str(snapshot.relative_to(PROJECT_ROOT))] = expected
        input_manifest["preserved_source_snapshot_sha256"] = preserved_sources
    else:
        for path in (model_root, prediction_root, report_root):
            path.mkdir(parents=True, exist_ok=False)
    write_json(execution_manifest_path, input_manifest)
    print(json.dumps({"status": "TRAINING", "run_id": run_id, "jobs": len(YEARS) * len(VARIANTS), "workers": workers}), flush=True)

    jobs = [(year, variant) for year in YEARS for variant in VARIANTS]
    receipts = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        for batch_start in range(0, len(jobs), workers):
            batch = jobs[batch_start:batch_start + workers]
            pending = {
                executor.submit(
                    train_job, str(research_root), str(model_root / variant / str(year)),
                    str(prediction_root / f"{variant}_{year}.parquet"), year, variant, seeds, device,
                    resume_checkpoints,
                ): (year, variant)
                for year, variant in batch
            }
            for future in as_completed(pending):
                year, variant = pending[future]
                receipts.append(future.result())
                print(json.dumps({"status": "ANNUAL_COMPLETE", "year": year, "variant": variant, "completed": len(receipts)}), flush=True)
    write_json(report_root / "training_jobs.json", {"jobs": receipts})

    for relative, expected in input_manifest["inputs"].items():
        if sha(research_root / relative) != expected:
            raise ValueError(f"a read-only research input changed during training: {relative}")
    for relative, expected in input_manifest["code"].items():
        if sha(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"experiment code changed during training: {relative}")
    for relative, expected in checkpoint_hashes.items():
        if sha(model_root / relative) != expected:
            raise ValueError(f"a reused component checkpoint changed: {relative}")
    if sha(protocol_path) != input_manifest["protocol_sha256"]:
        raise ValueError("preregistered protocol changed during training")

    predictions = pd.concat([
        pd.read_parquet(prediction_root / f"{variant}_{year}.parquet")
        for year, variant in jobs
    ], ignore_index=True)
    config = json.loads(config_path.read_text())
    baseline = pd.read_parquet(research_root / config["input"]["path"])
    baseline.golgg_match_id = baseline.golgg_match_id.astype(str)
    checks = check_probabilities(predictions, baseline)
    write_json(report_root / "numeric_checks.json", {"status": "PASS", "variants": checks})
    inference_checks = check_saved_inference(model_root, research_root, predictions)
    write_json(report_root / "inference_checks.json", {"status": "PASS", "models": inference_checks})
    benchmarks = {}
    for variant in VARIANTS:
        candidate_path = prediction_root / f"{variant}_test.parquet"
        predictions.loc[(predictions.variant == variant) & (predictions.partition == "test")].to_parquet(candidate_path, index=False)
        benchmarks[variant] = run_suite(
            config_path, research_root, report_root / f"benchmark_{variant}",
            candidate_data=candidate_path, candidate_col="p", draws=draws,
        )
    from src.analysis.a0_uncertainty import evaluate_uncertainty

    uncertainty = evaluate_uncertainty(predictions, baseline, report_root / "uncertainty", draws=draws)
    artifacts = {
        str(path.relative_to(PROJECT_ROOT)): sha(path)
        for directory in (model_root, prediction_root, report_root)
        for path in sorted(directory.rglob("*")) if path.is_file()
    }
    completed = {
        "status": "COMPLETE", "research_status": "DEVELOPMENT_ONLY", "run_id": run_id,
        "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
        "protocol_sha256": sha(protocol_path),
        "input_manifest_path": str(execution_manifest_path.relative_to(PROJECT_ROOT)),
        "input_manifest_sha256": sha(execution_manifest_path),
        "input_manifest_ancestry": input_manifest.get("parent_input_manifest"),
        "benchmarks": benchmarks, "uncertainty": uncertainty, "numeric_checks": checks,
        "resumption": resume_checkpoints,
        "artifact_sha256": artifacts,
        "inference_checks": inference_checks,
        "production_changed": False,
    }
    write_json(report_root / "completed.json", completed)
    print(json.dumps({"status": "COMPLETE", "run_id": run_id, "report": str(report_root / "completed.json"), "elapsed_seconds": completed["elapsed_seconds"]}), flush=True)
    return completed


def target_comparison_main(argv: list[str]) -> None:
    """Dispatch the isolated frozen native-target experiment workflow."""
    parser = argparse.ArgumentParser(
        description="Train or aggregate the frozen native A0 target comparison.",
    )
    commands = parser.add_subparsers(dest="action", required=True)
    train = commands.add_parser("train", help="fit one objective/year annual arm")
    train.add_argument("--protocol", type=Path, required=True)
    train.add_argument("--research-root", type=Path)
    train.add_argument("--objective", choices=("terminal", "winner"), required=True)
    train.add_argument("--year", type=int, choices=(2024, 2025, 2026), required=True)
    train.add_argument("--device", choices=("cpu",), default="cpu")
    train.add_argument("--resume", action="store_true")
    smoke = commands.add_parser("smoke", help="fit one seed for one epoch in a separate unscored namespace")
    smoke.add_argument("--protocol", type=Path, required=True)
    smoke.add_argument("--research-root", type=Path)
    smoke.add_argument("--objective", choices=("terminal", "winner"), required=True)
    smoke.add_argument("--year", type=int, choices=(2024, 2025, 2026), required=True)
    smoke.add_argument("--device", choices=("cpu",), default="cpu")
    aggregate = commands.add_parser("aggregate", help="export complete canonical candidates from annual TEST slices")
    aggregate.add_argument("--protocol", type=Path, required=True)
    aggregate.add_argument("--research-root", type=Path)
    aggregate.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    from src.models.a0_target_experiment import (
        aggregate_target_comparison,
        smoke_target_objective,
        train_annual_target_comparison,
    )
    if args.action == "train":
        result = train_annual_target_comparison(
            args.protocol, args.research_root, objective=args.objective,
            year=args.year, device=args.device, resume=args.resume,
        )
        output = {
            "status": result["status"], "objective": result["objective"],
            "year": result["year"], "prediction_path": result["prediction_path"],
            "prediction_rows": result["prediction_rows"],
        }
    elif args.action == "smoke":
        result = smoke_target_objective(
            args.protocol, args.research_root, objective=args.objective,
            year=args.year, device=args.device,
        )
        output = {
            "status": result["status"], "scoring_eligible": False,
            "objective": result["main_objective"], "year": result["year"],
            "train_rows": result["train_rows"],
        }
    else:
        result = aggregate_target_comparison(
            args.protocol, args.research_root, output_dir=args.output_dir,
        )
        output = {
            "status": result["status"], "objectives": result["objectives"],
            "canonical_rows": result["canonical_rows"],
            "candidate_schema": result["candidate_schema"],
            "artifacts_sha256": result["artifacts_sha256"],
        }
    print(json.dumps(output, ensure_ascii=False, allow_nan=False), flush=True)


def native_refit_main(argv: list[str]) -> None:
    """Dispatch the frozen native neural monthly-refit workflow."""
    parser = argparse.ArgumentParser(
        description="Run native-A0 monthly refits, bounded preflight, or candidate aggregation.",
    )
    commands = parser.add_subparsers(dest="action", required=True)
    smoke = commands.add_parser("smoke", help="fit one real-data epoch for both weighting paths")
    smoke.add_argument("--protocol", type=Path, required=True)
    smoke.add_argument("--research-root", type=Path)
    train = commands.add_parser("train", help="fit all 45 monthly origins for one weighting")
    train.add_argument("--protocol", type=Path, required=True)
    train.add_argument("--research-root", type=Path)
    train.add_argument("--weighting", choices=("uniform", "h365"), required=True)
    train.add_argument("--resume", action="store_true")
    aggregate = commands.add_parser("aggregate", help="calibrate and export all four canonical candidates")
    aggregate.add_argument("--protocol", type=Path, required=True)
    aggregate.add_argument("--research-root", type=Path)
    aggregate.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    from src.models.a0_native_refit import aggregate as aggregate_native
    from src.models.a0_native_refit import smoke as smoke_native
    from src.models.a0_native_refit import train_weighting

    if args.action == "smoke":
        result = smoke_native(args.protocol, args.research_root)
        output = {
            "status": result["status"],
            "scoring_eligible": False,
            "preflights": {
                weighting: {
                    "rows": fit["rows"],
                    "checkpoint": fit["checkpoint"],
                    "fit_seconds": fit["fit_seconds"],
                }
                for weighting, fit in result["preflights"].items()
            },
        }
    elif args.action == "train":
        result = train_weighting(
            args.protocol, args.weighting, args.research_root, resume=args.resume,
        )
        output = {
            "status": result["status"], "weighting": result["weighting"],
            "origins": result["origins"], "seed_fits": result["seed_fits"],
            "training_receipt": result["training_receipt"],
            "model_root": result["model_root"], "forecast_root": result["forecast_root"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
    else:
        result = aggregate_native(args.protocol, args.research_root, resume=args.resume)
        output = {
            "status": result["status"], "canonical_rows": result["canonical_rows"],
            "candidate_paths": result["candidate_paths"],
            "aggregate_receipt": result["aggregate_receipt"],
            "saved_inference_verification": {
                name: receipt["status"]
                for name, receipt in result["saved_inference_verification"].items()
            },
        }
    print(json.dumps(output, ensure_ascii=False, allow_nan=False), flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "native-refit":
        native_refit_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "target-comparison":
        target_comparison_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="Previously written experiment protocol JSON.")
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume-checkpoints", action="store_true", help="Resume this partial run from verified annual checkpoints.")
    args = parser.parse_args()
    root = args.research_root or os.environ.get("ENSEMBLE_RESEARCH_ROOT")
    pointer = PROJECT_ROOT / "data/research_root.txt"
    if not root and pointer.is_file():
        root = pointer.read_text().strip()
    if not root:
        parser.error("provide --research-root, ENSEMBLE_RESEARCH_ROOT or data/research_root.txt")
    run(
        args.protocol.resolve(), Path(root).expanduser().resolve(),
        args.workers, args.device, args.resume_checkpoints,
    )


if __name__ == "__main__":
    main()
