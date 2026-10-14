#!/usr/bin/env python3
"""EXP-085: chronological OOF rating stacking, local retrospective research only.

Run from repository root (no project dependencies are changed)::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
        scripts/experiment_rating_stack.py --run-name semester-20260908-a

Each run directory is exclusive-create. All outer/inner fitted estimators, raw
OOF and test predictions, calibration slopes, effective weights and provenance
are retained. No production registry, database, or frozen artifact is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_model_redesign import align_legacy_context, temporal_blocks
from scripts.benchmark_siamese_architectures import (
    monthly_bootstrap,
    probability_metrics,
)
from scripts.build_siamese_research_dataset import sha256
from scripts.train_and_tune_siamese_series import (
    build_training_features,
    fit_platt_scaling,
)
from src.analysis.probability_metrics import binary_log_loss_vector

SEED = 83
FAMILIES = ("elo", "gl", "ts", "os", "pl", "tm")
EPS = np.finfo(float).eps
LEGACY = ROOT / "data/artifacts/siamese-integrity-v1/snapshots.csv"
REPLAY = ROOT / "data/artifacts/model-redesign-v1/replay/snapshots.csv"
BENCHMARK = (
    ROOT
    / "data/artifacts/model-redesign-v1/benchmark-semester/legacy79/predictions.csv"
)
VARIANTS = (
    "baseline79",
    "rs_player_w20",
    "rs_joint_ridge",
    "rs_joint_convex",
    "rs_player_ridge",
    "rs_player_convex",
)


def open_probability(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid predicted probabilities")
    return np.clip(p, EPS, 1 - EPS)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def bounds(frame, mask):
    dates = frame.loc[mask, "date"]
    return {
        "n": len(dates),
        "min": str(dates.min().date()),
        "max": str(dates.max().date()),
    }


def fit_ridge(x, y, names):
    # Column-subset fancy indexing creates Fortran order; preserve the frozen
    # canonical control's C-order reductions and optimizer trajectory.
    x = np.ascontiguousarray(x)
    scaler = StandardScaler(with_mean=False).fit(x)
    model = LogisticRegression(
        C=0.1,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-8,
        random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    return {"scaler": scaler, "model": model, "names": list(names)}


def ridge_logits(model, x):
    return model["model"].decision_function(
        model["scaler"].transform(np.ascontiguousarray(x))
    )


def ridge_metadata(model):
    return {
        "feature_names": model["names"],
        "scale": model["scaler"].scale_.tolist(),
        "coefficients": model["model"].coef_[0].tolist(),
        "iterations": int(model["model"].n_iter_[0]),
        "C": 0.1,
        "intercept": False,
    }


def stream_indices(names, side):
    """Partition canonical odd inputs; no global consensus shared across families."""
    streams = {}
    for family in FAMILIES:
        # Canonical raw Glicko differences use `glicko`, probabilities use `gl`.
        family_names = ("gl", "glicko") if family == "gl" else (family,)
        sides = ("team", "player") if side == "joint" else ("player",)
        prefixes = tuple(f"{s}_{f}_" for s in sides for f in family_names)
        streams[family] = np.array(
            [i for i, name in enumerate(names) if name.startswith(prefixes)], dtype=int
        )
    streams["w20_rest"] = np.array(
        [
            i
            for i, name in enumerate(names)
            if name.startswith("w20_") or name == "rest_days_diff"
        ],
        dtype=int,
    )
    if any(len(indices) == 0 for indices in streams.values()):
        raise ValueError("empty family stream")
    expected = {
        i
        for i, name in enumerate(names)
        if name.startswith(
            ("team_", "player_", "w20_") if side == "joint" else ("player_", "w20_")
        )
        or name == "rest_days_diff"
    }
    assigned = [int(i) for indices in streams.values() for i in indices]
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError(
            "family partition loses or duplicates canonical raw/uncertainty inputs"
        )
    return streams


def fit_bases(x, y, names, streams):
    return {
        name: fit_ridge(x[:, indices], y, [names[i] for i in indices])
        for name, indices in streams.items()
    }


def base_logits(bases, x, streams):
    return np.column_stack(
        [ridge_logits(bases[name], x[:, indices]) for name, indices in streams.items()]
    )


def fit_convex(z, y):
    probabilities = open_probability(expit(z))

    def loss(w):
        p = open_probability(probabilities @ w)
        return float(binary_log_loss_vector(y, p).mean())

    def gradient(w):
        p = open_probability(probabilities @ w)
        return probabilities.T @ ((p - y) / (p * (1 - p))) / len(y)

    result = minimize(
        loss,
        np.full(z.shape[1], 1 / z.shape[1]),
        jac=gradient,
        method="SLSQP",
        bounds=[(0, 1)] * z.shape[1],
        constraints={
            "type": "eq",
            "fun": lambda w: w.sum() - 1,
            "jac": lambda w: np.ones_like(w),
        },
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success or abs(result.x.sum() - 1) > 1e-8:
        raise ValueError(f"convex meta optimization failed: {result.message}")
    return {
        "weights": result.x,
        "iterations": int(result.nit),
        "oof_training_log_loss": float(result.fun),
    }


def meta_logits(meta, z, kind):
    if kind == "ridge":
        return ridge_logits(meta, z)
    return logit(open_probability(expit(z) @ meta["weights"]))


def redundancy(z, names):
    """Correlation and participation-ratio effective rank on OOF only."""
    correlation = np.corrcoef(z, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(correlation), 0)
    off_diagonal = correlation[np.triu_indices(len(names), 1)]
    return {
        "source": "chronological train-only OOF logits, never outer test",
        "stream_names": list(names),
        "correlation": correlation.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "effective_rank_participation": float(
            eigenvalues.sum() ** 2 / (eigenvalues**2).sum()
        ),
        "mean_off_diagonal_correlation": float(off_diagonal.mean()),
        "min_off_diagonal_correlation": float(off_diagonal.min()),
        "max_off_diagonal_correlation": float(off_diagonal.max()),
    }


def effective_weights(meta, bases, streams, names, x_train, z_oof, kind, slope):
    if kind == "convex":
        w = meta["weights"]
        return {
            "probability_blend_weights": dict(zip(streams, w.tolist())),
            "inverse_weight_concentration": float(1 / np.dot(w, w)),
            "note": "convex probability mixture has no constant canonical logit coefficient; H2 slope is separate",
        }
    w = meta["model"].coef_[0] / meta["scaler"].scale_
    combined = np.zeros(len(names))
    for j, (name, indices) in enumerate(streams.items()):
        base = bases[name]
        combined[indices] += w[j] * base["model"].coef_[0] / base["scaler"].scale_
    combined *= slope
    np.testing.assert_allclose(
        slope * ridge_logits(meta, base_logits(bases, x_train, streams)),
        x_train @ combined,
        atol=1e-10,
        rtol=1e-10,
    )
    contributions = np.abs(combined * x_train.std(axis=0))
    total = contributions.sum()
    groups = {
        prefix: float(
            contributions[
                [i for i, name in enumerate(names) if name.startswith(prefix)]
            ].sum()
            / total
        )
        for prefix in ("team_", "player_", "w20_", "rest_days_diff")
    }
    return {
        "stream_logit_coefficients": dict(zip(streams, w.tolist())),
        "stream_oof_scaled_coefficients": dict(
            zip(streams, (w * z_oof.std(axis=0)).tolist())
        ),
        "canonical_effective_calibrated_coefficients": dict(
            zip(names, combined.tolist())
        ),
        "absolute_train_std_scaled_share": groups,
        "note": "signed/correlated coefficients are not independent importance or causal effects",
    }


def paired(frame, candidate, reference):
    y = frame.y_true.to_numpy()
    p, q = frame[candidate].to_numpy(), frame[reference].to_numpy()
    return {
        "log_loss": monthly_bootstrap(
            binary_log_loss_vector(y, p) - binary_log_loss_vector(y, q),
            frame.date,
            5000,
        ),
        "brier": monthly_bootstrap((p - y) ** 2 - (q - y) ** 2, frame.date, 5000),
    }


def execute(output):
    started = time.monotonic()
    sources = [
        LEGACY,
        REPLAY,
        BENCHMARK,
        Path(__file__),
        ROOT / "scripts/benchmark_model_redesign.py",
        ROOT / "scripts/benchmark_siamese_architectures.py",
        ROOT / "scripts/train_and_tune_siamese_series.py",
        ROOT / "src/models/symmetric_series.py",
        ROOT / "src/analysis/probability_metrics.py",
        ROOT / "src/analysis/metrics.py",
    ]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    specification = {
        "experiment": "EXP-085",
        "variants": list(VARIANTS),
        "seed": SEED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_code_sha256": source_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                p: version(p)
                for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
            },
        },
        "outer_years": [2024, 2025, 2026],
        "protocol": "semester-calibration",
        "inner": "predict year K from strictly earlier 2020..K-1; K=2021..Y-2; meta fits concatenated original-side OOF only",
        "training": {
            "base_C": 0.1,
            "meta_C": 0.1,
            "solver": "lbfgs",
            "max_iter": 10000,
            "tol": 1e-8,
            "convex_solver": "SLSQP",
            "convex_ftol": 1e-12,
            "convex_maxiter": 2000,
            "scaling": "train-only StandardScaler(with_mean=False)",
            "hyperparameter_search": "none, predeclared fixed controls and two meta forms",
            "stop_select": "reserved and diagnosed, never used to fit parameters or choose reported variants",
            "calibration": "independent positive zero-intercept Platt slope, outer H2 only",
        },
        "limitations": [
            "retrospective/exploratory: already-inspected 2024+ holdout",
            "legacy rating provenance and EXP081 cutoff UNVERIFIED",
            "inherited legacy canonical snapshots: pre-match/date freezing not newly established by stacking",
            "no closing odds, target draft/patch/current roster/current outcome used as features",
            "no reliability interaction: no claim of independent reliability evidence",
            "no promotion or live-profit claim",
        ],
    }
    write_json(output / "specification.json", specification)
    dtype = {"golgg_match_id": str, "team1_id": str, "team2_id": str}
    legacy, replay = (
        pd.read_csv(p, dtype=dtype, parse_dates=["date"]) for p in (LEGACY, REPLAY)
    )
    full, alignment = align_legacy_context(legacy, replay)
    full = full.sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    benchmark = pd.read_csv(
        BENCHMARK,
        dtype=dtype,
        parse_dates=["date"],
        usecols=[*dtype, "date", "y_true", "best_of", "fixed_ridge"],
    )
    benchmark = benchmark.sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    x, names = build_training_features(full)
    if x.shape[1] != 79 or not np.isfinite(x).all():
        raise ValueError("canonical79 input contract failed")
    y = full.y_true.to_numpy(dtype=int)
    models_dir = output / "models"
    models_dir.mkdir()
    year_metadata, predictions = {}, []
    for year in (2024, 2025, 2026):
        masks = temporal_blocks(full.date, year, protocol="semester-calibration")
        if not all(mask.any() for mask in masks.values()):
            raise ValueError(f"empty outer stage for {year}")
        train, cal, test = (masks[k] for k in ("train", "calibration", "test"))
        for earlier, later in zip(masks, list(masks)[1:]):
            if (
                full.loc[masks[earlier], "date"].max()
                >= full.loc[masks[later], "date"].min()
            ):
                raise ValueError("outer date boundary overlap")
        part = full.loc[
            test, ["golgg_match_id", "date", "y_true", "best_of", "competition_tier"]
        ].copy()
        ym = {
            "stages": {k: bounds(full, m) for k, m in masks.items()},
            "controls": {},
            "stacks": {},
        }
        expected = benchmark.loc[benchmark.date.dt.year == year].reset_index(drop=True)
        actual = full.loc[test].reset_index(drop=True)
        for column in (
            "golgg_match_id",
            "date",
            "y_true",
            "best_of",
            "team1_id",
            "team2_id",
        ):
            if not actual[column].equals(expected[column]):
                raise ValueError(f"common cohort/order/side mismatch: {year} {column}")
        for variant, indices in [
            ("baseline79", np.arange(len(names))),
            (
                "rs_player_w20",
                np.array(
                    [
                        i
                        for i, n in enumerate(names)
                        if n.startswith(("player_", "w20_")) or n == "rest_days_diff"
                    ]
                ),
            ),
        ]:
            model = fit_ridge(
                x[train][:, indices], y[train], [names[i] for i in indices]
            )
            z_cal = logit(
                open_probability(expit(ridge_logits(model, x[cal][:, indices])))
            )
            slope = fit_platt_scaling(z_cal, y[cal])
            z = logit(open_probability(expit(ridge_logits(model, x[test][:, indices]))))
            p = open_probability(expit(slope * z))
            reverse = open_probability(
                expit(
                    slope
                    * logit(
                        open_probability(
                            expit(ridge_logits(model, -x[test][:, indices]))
                        )
                    )
                )
            )
            symmetry = float(np.max(np.abs(p + reverse - 1)))
            if symmetry > 1e-6:
                raise ValueError("control symmetry failed")
            part[f"p__{variant}"] = p
            part[f"raw__{variant}"] = open_probability(expit(z))
            ym["controls"][variant] = {
                **ridge_metadata(model),
                "calibration_slope": slope,
                "symmetry_max_error": symmetry,
            }
            joblib.dump(
                {"model": model, "indices": indices, "calibration_slope": slope},
                models_dir / f"{year}-{variant}.joblib",
            )
            if variant == "baseline79":
                np.testing.assert_allclose(p, expected.fixed_ridge, atol=1e-10, rtol=0)
                ym["baseline_frozen_max_error"] = float(
                    np.max(np.abs(p - expected.fixed_ridge.to_numpy()))
                )
        for side in ("joint", "player"):
            streams = stream_indices(names, side)
            oof = np.full((len(full), len(streams)), np.nan)
            inner_folds = []
            for inner_year in range(2021, year - 1):
                inner_train = train & (full.date < f"{inner_year}-01-01").to_numpy()
                inner_predict = train & (full.date.dt.year == inner_year).to_numpy()
                if not inner_train.any() or not inner_predict.any():
                    raise ValueError("empty chronological inner fold")
                if (
                    full.loc[inner_train, "date"].max()
                    >= full.loc[inner_predict, "date"].min()
                ):
                    raise ValueError("inner chronological leakage")
                bases = fit_bases(x[inner_train], y[inner_train], names, streams)
                oof[inner_predict] = base_logits(bases, x[inner_predict], streams)
                inner_folds.append(
                    {
                        "prediction_year": inner_year,
                        "fit": bounds(full, inner_train),
                        "predict": bounds(full, inner_predict),
                        "base_parameters": {
                            k: ridge_metadata(v) for k, v in bases.items()
                        },
                    }
                )
                joblib.dump(
                    {"bases": bases, "streams": streams},
                    models_dir / f"{year}-{side}-inner{inner_year}.joblib",
                )
            eligible = np.isfinite(oof).all(axis=1)
            if np.any(eligible & ~train) or np.any(
                np.isfinite(oof).any(axis=1) & ~eligible
            ):
                raise ValueError("OOF assignment outside outer train or incomplete")
            oof_frame = full.loc[eligible, ["golgg_match_id", "date", "y_true"]].copy()
            for j, name in enumerate(streams):
                oof_frame[f"z__{name}"] = oof[eligible, j]
            oof_frame.to_csv(output / f"oof-{year}-{side}.csv", index=False)
            bases = fit_bases(x[train], y[train], names, streams)
            stage_z = {k: base_logits(bases, x[m], streams) for k, m in masks.items()}
            reverse_z = base_logits(bases, -x[test], streams)
            np.testing.assert_allclose(stage_z["test"], -reverse_z, atol=1e-12, rtol=0)
            sm = {
                "streams": {k: [names[i] for i in v] for k, v in streams.items()},
                "inner_folds": inner_folds,
                "oof_coverage": bounds(full, eligible),
                "train_rows_without_oof": int((train & ~eligible).sum()),
                "redundancy": redundancy(oof[eligible], streams),
                "bases": {k: ridge_metadata(v) for k, v in bases.items()},
                "meta": {},
            }
            for kind in ("ridge", "convex"):
                variant = f"rs_{side}_{kind}"
                meta = (
                    fit_ridge(oof[eligible], y[eligible], list(streams))
                    if kind == "ridge"
                    else fit_convex(oof[eligible], y[eligible])
                )
                z_cal = meta_logits(meta, stage_z["calibration"], kind)
                slope = fit_platt_scaling(z_cal, y[cal])
                z = meta_logits(meta, stage_z["test"], kind)
                p = open_probability(expit(slope * z))
                reverse = open_probability(
                    expit(slope * meta_logits(meta, reverse_z, kind))
                )
                symmetry = float(np.max(np.abs(p + reverse - 1)))
                if symmetry > 1e-6:
                    raise ValueError("stack symmetry failed")
                part[f"p__{variant}"] = p
                part[f"raw__{variant}"] = open_probability(expit(z))
                metadata = (
                    ridge_metadata(meta)
                    if kind == "ridge"
                    else {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in meta.items()
                    }
                )
                sm["meta"][kind] = {
                    "parameters": metadata,
                    "calibration_slope": slope,
                    "symmetry_max_error": symmetry,
                    "effective_weights": effective_weights(
                        meta,
                        bases,
                        streams,
                        names,
                        x[train],
                        oof[eligible],
                        kind,
                        slope,
                    ),
                    "reserved_stage_raw_diagnostics": {
                        k: probability_metrics(
                            y[masks[k]],
                            open_probability(
                                expit(meta_logits(meta, stage_z[k], kind))
                            ),
                        )
                        for k in ("stop", "select")
                    },
                }
                artifact = {
                    "version": "exp085-rating-stack-v1",
                    "canonical_names": names,
                    "streams": streams,
                    "bases": bases,
                    "meta": meta,
                    "kind": kind,
                    "calibration_slope": slope,
                }
                path = models_dir / f"{year}-{variant}.joblib"
                joblib.dump(artifact, path)
                loaded = joblib.load(path)
                restored_z = base_logits(loaded["bases"], x[test], loaded["streams"])
                restored_p = open_probability(
                    expit(
                        loaded["calibration_slope"]
                        * meta_logits(loaded["meta"], restored_z, loaded["kind"])
                    )
                )
                np.testing.assert_array_equal(p, restored_p)
            ym["stacks"][side] = sm
        year_metadata[str(year)] = ym
        write_json(output / f"year-{year}.json", ym)
        part.to_csv(output / f"predictions-{year}.csv", index=False)
        predictions.append(part)
        print(f"Completed {year}: test={test.sum()} train={train.sum()}", flush=True)
    prediction = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    if (
        prediction.golgg_match_id.duplicated().any()
        or prediction.golgg_match_id.tolist() != benchmark.golgg_match_id.tolist()
    ):
        raise ValueError("duplicate IDs or final cohort mismatch")
    prediction.to_csv(output / "predictions.csv", index=False)
    metrics, paired_baseline = {}, {}
    for variant in VARIANTS:
        column = f"p__{variant}"
        metrics[variant] = {
            "aggregate": probability_metrics(prediction.y_true, prediction[column]),
            "per_year": {
                str(year): probability_metrics(group.y_true, group[column])
                for year, group in prediction.groupby(prediction.date.dt.year)
            },
        }
        paired_baseline[variant] = {
            "aggregate": paired(prediction, column, "p__baseline79"),
            "per_year": {
                str(year): paired(group, column, "p__baseline79")
                for year, group in prediction.groupby(prediction.date.dt.year)
            },
        }
    team_increment = {
        kind: paired(prediction, f"p__rs_joint_{kind}", f"p__rs_player_{kind}")
        for kind in ("ridge", "convex")
    }
    team_increment["direct79_vs_player_w20"] = paired(
        prediction, "p__baseline79", "p__rs_player_w20"
    )
    artifacts = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    if source_hashes != {str(p.relative_to(ROOT)): sha256(p) for p in sources}:
        raise ValueError("source or input changed during run")
    summary = {
        **specification,
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "alignment": alignment,
        "coverage": {
            "expected_test_rows": len(benchmark),
            "actual_test_rows": len(prediction),
            "eligibility_loss": 0,
            "min": str(prediction.date.min().date()),
            "max": str(prediction.date.max().date()),
        },
        "feature_names": names,
        "years": year_metadata,
        "metrics": metrics,
        "paired_vs_baseline79": paired_baseline,
        "team_increment": team_increment,
        "artifact_sha256": artifacts,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {
            "input_code_sha256": source_hashes,
            "artifact_sha256": {
                **artifacts,
                "summary.json": sha256(output / "summary.json"),
            },
            "serialization": "joblib dictionaries with sklearn objects; load only trusted locally generated artifacts",
        },
    )
    print(
        json.dumps(
            {variant: value["aggregate"] for variant, value in metrics.items()},
            indent=2,
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be one directory component")
    output = ROOT / "data/artifacts/exp085-rating-stack" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    try:
        execute(output)
    except Exception:
        write_json(
            output / "failure.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
