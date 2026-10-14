"""EXP-090: locked exploratory full79 recency/residual/neural/prequential study.

Run with ``python -m scripts.experiment_adaptive_series run --output-dir PATH``.
All inputs are local immutable research CSVs; no database or market fitting.
"""

import argparse
import hashlib
import itertools
import json
import platform
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.adaptive_series_models import (
    LinearScore,
    deployed_logits,
    fit_prequential_slope,
    fit_residual_booster,
    fit_ridge,
    recency_weights,
)
from scripts.benchmark_model_redesign import align_legacy_context, temporal_blocks
from scripts.early_stopped_siamese import fit_early_stopped_ensemble
from scripts.evaluate_adaptive_series import evaluate
from scripts.train_and_tune_siamese_series import (
    build_training_features,
    fit_platt_scaling,
)
from src.models.redesign_series import SeriesEstimator, _log_loss, _open_probability

VERSION = "exp090-adaptive-series-v1"
INPUTS = {
    "legacy": "data/artifacts/siamese-integrity-v1/snapshots.csv",
    "replay": "data/artifacts/model-redesign-v1/replay/snapshots.csv",
    "reference": "data/artifacts/exp089-oddsfree-distillation/run01/predictions.csv",
    "control_2024": "data/artifacts/model-redesign-v1/benchmark-semester/legacy79/2024-fixed_ridge.joblib",
    "control_2025": "data/artifacts/model-redesign-v1/benchmark-semester/legacy79/2025-fixed_ridge.joblib",
    "control_2026": "data/artifacts/model-redesign-v1/benchmark-semester/legacy79/2026-fixed_ridge.joblib",
}
SOURCES = (
    "scripts/adaptive_series_models.py",
    "scripts/early_stopped_siamese.py",
    "scripts/experiment_adaptive_series.py",
    "scripts/evaluate_adaptive_series.py",
    "scripts/train_and_tune_siamese_series.py",
    "scripts/benchmark_model_redesign.py",
    "scripts/benchmark_siamese_architectures.py",
    "scripts/evaluate_oddsfree_distillation.py",
    "src/models/redesign_series.py",
    "src/models/symmetric_series.py",
    "src/analysis/metrics.py",
)


def _hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(path, payload):
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def _progress(stage, **details):
    print(json.dumps({"stage": stage, **details}, allow_nan=False), flush=True)


def _inputs(paths):
    legacy, replay, reference = (
        pd.read_csv(paths[key], float_precision="round_trip")
        for key in ("legacy", "replay", "reference")
    )
    frame, alignment = align_legacy_context(legacy, replay)
    if alignment["matched_rows"] != alignment["legacy_rows"]:
        raise ValueError("alignment would exclude historical state or evaluation rows")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    frame = frame.sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    if frame.date.isna().any() or frame.golgg_match_id.duplicated().any():
        raise ValueError("historical dates and unique series IDs are required")
    modern = frame.loc[frame.date >= "2024-01-01"]
    if reference.golgg_match_id.duplicated().any() or set(
        reference.golgg_match_id
    ) != set(modern.golgg_match_id):
        raise ValueError("reference and full modern legacy inventories differ")
    reference = (
        reference.set_index("golgg_match_id").loc[modern.golgg_match_id].reset_index()
    )
    if not np.array_equal(
        reference.y_true.to_numpy(), modern.y_true.to_numpy()
    ) or not np.array_equal(
        pd.to_datetime(reference.date).to_numpy(), modern.date.to_numpy()
    ):
        raise ValueError("reference date/target alignment differs")
    x, names = build_training_features(frame)
    if len(names) != 79:
        raise ValueError("experiment requires the entire canonical79 sports mapping")
    return frame, x, names, reference, alignment


def _probability(score, slope, x):
    return _open_probability(expit(slope * deployed_logits(score, x)))


def _save_model(output, stem, score, slope, names, metadata, records, report_column):
    path = output / f"{stem}.joblib"
    payload = {
        "model_version": VERSION,
        "feature_names": names,
        "score": score,
        "slope": float(slope),
        "metadata": metadata,
    }
    with path.open("xb") as handle:
        joblib.dump(payload, handle, compress=3)
    records.append(
        {
            "file": path.name,
            "sha256": _hash(path),
            "report_column": report_column,
            **metadata,
        }
    )


def _stage_audit(frame, masks):
    result = {}
    for name, mask in masks.items():
        dates = frame.loc[mask, "date"]
        if dates.empty:
            raise ValueError(f"empty {name} stage")
        result[name] = {
            "n": len(dates),
            "min": dates.min().isoformat(),
            "max": dates.max().isoformat(),
        }
    ordered = [
        result[key] for key in ("train", "stop", "select", "calibration", "test")
    ]
    if any(left["max"] >= right["min"] for left, right in itertools.pairwise(ordered)):
        raise ValueError(
            "annual stages are not disjoint whole-date chronological blocks"
        )
    return result


def run(output_dir: Path, paths=None):
    paths = dict(INPUTS if paths is None else paths)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    frame, x, names, reference, alignment = _inputs(paths)
    y = frame.y_true.to_numpy(float)
    modern_mask = (frame.date >= "2024-01-01").to_numpy()
    modern_indices = np.flatnonzero(modern_mask)
    metadata_columns = [column for column in reference if not column.startswith("p__")]
    predictions = reference[metadata_columns].copy()
    predictions["date"] = pd.to_datetime(predictions.date)
    predictions["p__stored_exp081_diagnostic"] = reference.p__stored_exp081_diagnostic
    source_hashes = {path: _hash(path) for path in SOURCES}
    protocol = {
        "experiment": "EXP-090",
        "model_version": VERSION,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "target": "P(team A wins complete series), all79 odd sports features; no odds inputs or teachers",
        "inputs": {
            key: {"path": path, "sha256": _hash(path)} for key, path in paths.items()
        },
        "code_sha256": source_hashes,
        "feature_names": names,
        "alignment": alignment,
        "evaluation_n": int(modern_mask.sum()),
        "evaluation_start": "2024-01-01",
        "annual": "TRAIN2020..Y-2; STOP Q1(Y-1); SELECT Q2(Y-1); CAL H2(Y-1); TEST Y; no refit",
        "control": "Exact frozen EXP083 C=.1 canonical79 ridge coefficients/scales/H2 slope; uniform_refit is a separate candidate, not claimed bit-identical to another solver run",
        "annual_candidates": [
            "outcome",
            "uniform_refit",
            "recency365",
            "residual_w25",
            "residual_w50",
            "residual_w100",
            "siamese_bce_early",
            "siamese_focal_early",
        ],
        "selection": "STOP-only provisional aggregate slope; Q2 LogLoss chooses residual weight and full annual family; independent H2 final aggregate slope",
        "recency": "365-day half-life, every TRAIN row retained; weights normalized to original total mass; same unweighted TRAIN scale and C=.1",
        "residual": "fixed full-TRAIN ridge offset; depth2/leaves4/minchild200/lambda100/lr.03/max500; STOP early stop50; weights0/.25/.5/1; all reported",
        "neural": "five bootstrap79-32-16-1 members; gamma0 and1; lr.001/decay.001/batch256/max60/patience8; calibrated aggregate STOP checkpoint; aggregate H2 slope",
        "prequential": "quarterly C=.1 ridge refit on all earlier dates since2020; uniform and365-day objectives; positive slope from prior two quarters' genuinely OOS raw logits, never final base's in-sample logits",
        "prequential_warmup": "2023Q3 and Q4; first scored quarter2024Q1; no annual selection imposed on fixed quarterly policies",
        "bootstrap": "paired monthly blocks5000; exploratory unadjusted CIs, not multiplicity-corrected confirmation",
        "eligibility": {
            "untouched_holdout": False,
            "corrected_feature_compatibility": False,
            "production_promotion": False,
            "source_and_roster_availability_verified": False,
        },
        "limitations": [
            "The entire historical cohort has been inspected repeatedly.",
            "Legacy rating inputs are not the corrected prospective simultaneous-v1 contract.",
            "Strict raw replay is blocked by contradictory results and corrupt identities; no historical transitions are silently discarded here.",
            "Stored EXP081 provenance is unverified and its declared training overlaps2024; diagnostic only.",
            "No EXP039 or active-operational PIT baseline is available here; required promotion comparisons are missing.",
            "Calendar dates are not verified source publication or real placement timestamps; no executable betting claims.",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                name: version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "lightgbm",
                    "joblib",
                )
            },
        },
    }
    _json(output / "protocol.json", protocol)
    for source in SOURCES:
        destination = output / "source_snapshot" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(Path(source).read_bytes())
    _progress("protocol_locked", n=len(predictions), feature_count=len(names))
    records, fold_audits = [], []
    candidate_arrays = {
        name: np.full(len(frame), np.nan)
        for name in protocol["annual_candidates"]
        + ["residual_selected", "annual_selected"]
    }
    for year in sorted(frame.loc[modern_mask, "date"].dt.year.unique()):
        year = int(year)
        masks = temporal_blocks(frame.date, year, "semester-calibration")
        stages = _stage_audit(frame, masks)
        train, stop, select, calibration, test = (
            masks[key] for key in ("train", "stop", "select", "calibration", "test")
        )
        xt, xs, xv, xc, xe = (
            x[mask] for mask in (train, stop, select, calibration, test)
        )
        yt, ys, yv, yc = (y[mask] for mask in (train, stop, select, calibration))
        _progress("annual_start", year=year, stages=stages)
        frozen = joblib.load(paths[f"control_{year}"])
        if (
            not isinstance(frozen, SeriesEstimator)
            or frozen.kind != "linear"
            or frozen.metadata["candidate"] != "fixed_ridge"
            or frozen.n_odd != len(names)
            or tuple(frozen.feature_names[: len(names)])
            != tuple(f"d_{name}" for name in names)
            or frozen.basis.context_indices
            or frozen.basis.spline is not None
            or frozen.base.C != 0.1
            or frozen.base.fit_intercept
            or frozen.metadata["stage_rows"]
            != {
                stage: stages[stage]["n"]
                for stage in ("train", "stop", "select", "calibration")
            }
        ):
            raise ValueError(
                "frozen control does not match the full79 annual fit contract"
            )
        ridge = LinearScore(frozen.basis.scaler.scale_, frozen.base.coef_[0])
        ridge_audit = {
            "source": paths[f"control_{year}"],
            "source_sha256": _hash(paths[f"control_{year}"]),
            "calibration": "saved independent H2 slope, not refitted",
        }
        refitted, refit_audit = fit_ridge(xt, yt)
        weights, weight_audit = recency_weights(
            frame.loc[train, "date"], f"{year - 1}-01-01"
        )
        recency, recency_audit = fit_ridge(xt, yt, sample_weight=weights)
        residual, residual_audit = fit_residual_booster(
            ridge, xt, yt, xs, ys, seed=90 + year
        )
        models = {"outcome": ridge, "uniform_refit": refitted, "recency365": recency}
        audits = {
            "outcome": ridge_audit,
            "uniform_refit": refit_audit,
            "recency365": {**recency_audit, "weights": weight_audit},
            "residual": residual_audit,
        }
        audits["residual"]["feature_gain"] = dict(
            zip(
                names,
                residual.booster.feature_importance(importance_type="gain").tolist(),
            )
        )
        for weight in (0.25, 0.5, 1.0):
            models[f"residual_w{int(weight * 100)}"] = replace(residual, weight=weight)
        for label, gamma in (("bce", 0.0), ("focal", 1.0)):
            _progress("neural_start", year=year, loss=label)
            score, audit = fit_early_stopped_ensemble(
                xt, yt, xs, ys, gamma=gamma, seed=90000 + year
            )
            models[f"siamese_{label}_early"] = score
            audits[f"siamese_{label}_early"] = audit
            _progress(
                "neural_fitted",
                year=year,
                loss=label,
                best_epoch=audit["best_epoch"],
                epochs=audit["epochs_ran"],
            )
        selection = {}
        for name, score in models.items():
            provisional = fit_platt_scaling(deployed_logits(score, xs), ys)
            selection[name] = {
                "provisional_stop_slope": provisional,
                "select_log_loss": _log_loss(yv, _probability(score, provisional, xv)),
            }
        winner = min(models, key=lambda name: selection[name]["select_log_loss"])
        residual_winner = min(
            ("outcome", "residual_w25", "residual_w50", "residual_w100"),
            key=lambda name: selection[name]["select_log_loss"],
        )
        models["annual_selected"] = models[winner]
        models["residual_selected"] = models[residual_winner]
        slopes = {}
        for name, score in models.items():
            slope = (
                frozen.slope
                if score is ridge
                else fit_platt_scaling(deployed_logits(score, xc), yc)
            )
            slopes[name] = slope
            candidate_arrays[name][test] = _probability(score, slope, xe)
            if name in audits:
                audits[name]["diagnostic_train_calibrated_log_loss"] = _log_loss(
                    yt, _probability(score, slope, xt)
                )
                audits[name]["diagnostic_calibration_log_loss"] = _log_loss(
                    yc, _probability(score, slope, xc)
                )
            metadata = {
                "policy": "annual",
                "candidate": name,
                "year": year,
                "stages": stages,
                "forecast_start": f"{year}-01-01",
                "forecast_end": f"{year + 1}-01-01",
                "fit_before": f"{year - 1}-01-01",
                "train_max_date": stages["train"]["max"],
                "calibration_max_date": stages["calibration"]["max"],
                "selection": winner
                if name == "annual_selected"
                else residual_winner
                if name == "residual_selected"
                else None,
            }
            _save_model(
                output,
                f"{year}-{name}",
                score,
                slope,
                names,
                metadata,
                records,
                f"p__{name}",
            )
        fold = {
            "year": year,
            "stages": stages,
            "selection_scores": selection,
            "annual_selected": winner,
            "residual_selected": residual_winner,
            "slopes": slopes,
            "fits": audits,
        }
        fold_audits.append(fold)
        _json(output / f"{year}-fold.json", fold)
        _progress(
            "annual_complete", year=year, selected=winner, residual=residual_winner
        )
    for name, values in candidate_arrays.items():
        predictions[f"p__{name}"] = values[modern_indices]
    control_difference = float(
        np.max(np.abs(predictions.p__outcome - reference.p__frozen_ridge))
    )
    if control_difference > 1e-12:
        raise ValueError(
            f"frozen control inference differs from frozen reference by {control_difference}"
        )
    _progress("control_reproduced", maximum_probability_difference=control_difference)
    oof_rows = []
    final_quarter = frame.date.max().to_period("Q").start_time
    quarters = pd.date_range("2023-07-01", final_quarter, freq="QS")
    for policy in ("quarterly_uniform", "quarterly_recency365"):
        raw_history = []
        values = np.full(len(frame), np.nan)
        for start in quarters:
            end = start + pd.DateOffset(months=3)
            train = ((frame.date >= "2020-01-01") & (frame.date < start)).to_numpy()
            test = ((frame.date >= start) & (frame.date < end)).to_numpy()
            if not train.any() or not test.any():
                raise ValueError(
                    "quarterly policy has an empty training or forecast cohort"
                )
            kwargs, weight_audit = {}, None
            if policy == "quarterly_recency365":
                kwargs["sample_weight"], weight_audit = recency_weights(
                    frame.loc[train, "date"], start
                )
            score, fit_audit = fit_ridge(x[train], y[train], **kwargs)
            raw = score.logit(x[test])
            current = frame.loc[test, ["golgg_match_id", "date", "y_true"]].copy()
            current["base_fit_before"] = start
            current["base_train_max_date"] = frame.loc[train, "date"].max()
            current["raw_logit"] = raw
            current["policy"] = policy
            slope, calibration_audit = 1.0, None
            scored = start >= pd.Timestamp("2024-01-01")
            if scored:
                slope, calibration_audit = fit_prequential_slope(
                    pd.concat(raw_history, ignore_index=True), start
                )
                values[test] = _probability(score, slope, x[test])
            raw_history.append(current)
            oof_rows.append(current)
            metadata = {
                "policy": policy,
                "candidate": policy,
                "forecast_start": start.isoformat(),
                "forecast_end": end.isoformat(),
                "fit_before": start.isoformat(),
                "train_max_date": frame.loc[train, "date"].max().isoformat(),
                "calibration_max_date": calibration_audit["latest_outcome_date"]
                if calibration_audit
                else None,
                "fit": fit_audit,
                "weights": weight_audit,
                "calibration": calibration_audit,
                "warmup_uncalibrated": not scored,
            }
            _save_model(
                output,
                f"{start.date()}-{policy}",
                score,
                slope,
                names,
                metadata,
                records,
                f"p__{policy}" if scored else None,
            )
            _progress(
                "quarter_complete",
                policy=policy,
                forecast_start=str(start.date()),
                n=int(test.sum()),
                slope=slope,
                calibration_n=calibration_audit["n"] if calibration_audit else None,
            )
        predictions[f"p__{policy}"] = values[modern_indices]
    predictions.to_csv(output / "predictions.csv", index=False, mode="x")
    pd.concat(oof_rows, ignore_index=True).to_csv(
        output / "prequential_oof.csv", index=False, mode="x"
    )
    _json(output / "models.json", records)
    _json(
        output / "fit_summary.json",
        {
            "folds": fold_audits,
            "control_maximum_probability_difference": control_difference,
        },
    )
    result = evaluate(predictions, output / "evaluation")
    proof = verify(output)
    _json(output / "verification.json", proof)
    _progress(
        "complete",
        n=result["n"],
        aggregate={
            key: {
                field: metrics.get(field)
                for field in ("log_loss", "brier", "ece10", "calibration_slope")
            }
            for key, metrics in result["aggregate"].items()
        },
        verification=proof,
    )
    return result


def verify(run_dir: Path):
    output = Path(run_dir)
    protocol = json.loads((output / "protocol.json").read_text())
    for source, digest in protocol["code_sha256"].items():
        if (
            _hash(source) != digest
            or _hash(output / "source_snapshot" / source) != digest
        ):
            raise ValueError(f"experiment code changed: {source}")
    paths = {}
    for name, entry in protocol["inputs"].items():
        if _hash(entry["path"]) != entry["sha256"]:
            raise ValueError(f"experiment input changed: {name}")
        paths[name] = entry["path"]
    frame, x, names, reference, alignment = _inputs(paths)
    predictions = pd.read_csv(
        output / "predictions.csv", float_precision="round_trip"
    ).set_index("golgg_match_id")
    oof = pd.read_csv(output / "prequential_oof.csv", float_precision="round_trip")
    records = json.loads((output / "models.json").read_text())
    maximum_error, maximum_symmetry, checked = 0.0, 0.0, 0
    for record in records:
        path = output / record["file"]
        if _hash(path) != record["sha256"]:
            raise ValueError(f"model artifact changed: {path}")
        artifact = joblib.load(
            path
        )  # Trusted exclusively generated local artifact only.
        if artifact["model_version"] != VERSION or artifact["feature_names"] != names:
            raise ValueError("incompatible model artifact or feature order")
        mask = (
            (frame.date >= record["forecast_start"])
            & (frame.date < record["forecast_end"])
        ).to_numpy()
        if pd.Timestamp(record["train_max_date"]) >= pd.Timestamp(
            record["forecast_start"]
        ):
            raise ValueError("model consumed a target-period outcome")
        if record["calibration_max_date"] and pd.Timestamp(
            record["calibration_max_date"]
        ) >= pd.Timestamp(record["forecast_start"]):
            raise ValueError("calibrator consumed a target-period outcome")
        actual = _probability(artifact["score"], artifact["slope"], x[mask])
        reverse = _probability(artifact["score"], artifact["slope"], -x[mask])
        maximum_symmetry = max(
            maximum_symmetry, float(np.max(np.abs(actual + reverse - 1)))
        )
        if record["report_column"]:
            expected = predictions.loc[
                frame.loc[mask, "golgg_match_id"], record["report_column"]
            ].to_numpy()
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            checked += len(actual)
        if record["policy"].startswith("quarterly_"):
            subset = oof.loc[
                (oof.policy == record["policy"])
                & (
                    pd.to_datetime(oof.base_fit_before)
                    == pd.Timestamp(record["forecast_start"])
                )
            ].set_index("golgg_match_id")
            expected_raw = subset.loc[
                frame.loc[mask, "golgg_match_id"], "raw_logit"
            ].to_numpy()
            np.testing.assert_allclose(
                artifact["score"].logit(x[mask]), expected_raw, rtol=0, atol=1e-12
            )
    if maximum_symmetry > 1e-6:
        raise ValueError("series probability symmetry failed")
    np.testing.assert_allclose(
        predictions.loc[reference.golgg_match_id, "p__outcome"].to_numpy(),
        reference.p__frozen_ridge.to_numpy(),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        predictions.loc[
            reference.golgg_match_id, "p__stored_exp081_diagnostic"
        ].to_numpy(),
        reference.p__stored_exp081_diagnostic.to_numpy(),
    )
    return {
        "verified": True,
        "model_artifacts": len(records),
        "reconstructed_predictions": checked,
        "maximum_probability_error": maximum_error,
        "maximum_symmetry_error": maximum_symmetry,
        "input_hashes_verified": True,
        "source_hashes_verified": True,
        "prequential_raw_oof_reconstructed": True,
        "legacy_source_availability_verified": False,
        "corrected_contract_compatible": False,
        "promotion": False,
        "alignment": alignment,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--output-dir", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.output_dir)
    else:
        print(json.dumps(verify(args.run_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
