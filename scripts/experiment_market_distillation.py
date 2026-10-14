#!/usr/bin/env python3
"""EXP-087: local, retrospective devig and sports-only soft-label distillation.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python
scripts/experiment_market_distillation.py --output-dir
 data/artifacts/exp087-market-distillation/<new-run-name>

No network/DB access. Frozen inputs and production artifacts are never modified.
Devig is selected only on Q2 of the prior year. All fixed teacher weights and
methods are reported, not ranked for promotion using the inspected holdout.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_model_redesign import align_legacy_context, temporal_blocks
from scripts.benchmark_siamese_architectures import (
    attach_market,
    monthly_bootstrap,
    probability_metrics,
)
from scripts.build_siamese_research_dataset import sha256
from src.analysis.probability_metrics import binary_log_loss_vector
from scripts.train_and_tune_siamese_series import (
    build_training_features,
    fit_platt_scaling,
)

METHODS = ("multiplicative", "power", "additive")
WEIGHTS = (0.1, 0.3, 0.5)
EPS = np.finfo(float).eps
SOURCE = "https://doi.org/10.11648/j.ajss.20170506.12"


def devig(odds_a, odds_b):
    """Return un-clipped binary probabilities; invalid prices remain missing."""
    a, b = np.asarray(odds_a, float), np.asarray(odds_b, float)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 1) & (b > 1)
    out = {name: np.full(len(a), np.nan) for name in METHODS}
    q1, q2 = 1 / a[valid], 1 / b[valid]
    sums = q1 + q2
    out["multiplicative"][valid] = q1 / sums
    additive = q1 - (sums - 1) / 2
    additive_ok = (additive > 0) & (additive < 1)
    out["additive"][np.flatnonzero(valid)[additive_ok]] = additive[additive_ok]
    powers = np.empty(len(q1))
    for i, (qa, qb) in enumerate(zip(q1, q2)):
        # Positive exponent exists for every pair of finite decimal odds > 1.
        upper = 1.0
        while qa**upper + qb**upper > 1:
            upper *= 2
        k = brentq(lambda k: qa**k + qb**k - 1, 0, upper, xtol=1e-13)
        powers[i] = qa**k
    out["power"][valid] = powers
    for name, p in out.items():
        finite = np.isfinite(p)
        if np.any((p[finite] <= 0) | (p[finite] >= 1)):
            raise ValueError(f"unbounded devig: {name}")
    # Shin equations: q_i^2 / S = (1-z)p_i^2 + z*p_i.
    # Their binary difference gives p1-p2=q1-q2, hence additive.
    over = (sums >= 1) & additive_ok
    p = additive[over]
    z = (q1[over] ** 2 / sums[over] - p**2) / (p - p**2)
    residual = (1 - z) * (1 - p) ** 2 + z * (1 - p) - q2[over] ** 2 / sums[over]
    shin_error = float(np.max(abs(residual))) if len(residual) else 0.0
    if shin_error > 1e-10 or np.any((z < -1e-10) | (z >= 1)):
        raise ValueError("binary Shin equivalence failed")
    return out, {
        "price_pairs": len(a),
        "valid_prices": int(valid.sum()),
        "invalid_or_missing_prices": int((~valid).sum()),
        "underround": int((sums < 1).sum()),
        "fair_sum": int((sums == 1).sum()),
        "overround": int((sums > 1).sum()),
        "invalid_additive_excluded_not_clipped": int((~additive_ok).sum()),
        "margin_quantiles": dict(
            zip(
                ("min", "q01", "median", "q99", "max"),
                map(float, np.quantile(sums - 1, [0, 0.01, 0.5, 0.99, 1])),
            )
        )
        if len(sums)
        else {},
        "shin_equivalence_n": int(over.sum()),
        "shin_equation_max_residual": shin_error,
        "underround_policy": "reported in direct market diagnostics; excluded from all matched student training controls",
    }


def train_student(x, target, c=0.1):
    """Actual Bernoulli soft-target BCE + L2, no hard-label estimator abuse.

    Sum BCE + ||w||²/(2C) matches the hard-label ridge convention. No intercept
    and no centering ensure odd logits and exact side symmetry. Numerical
    convergence uses only TRAIN; no validation-based stopping or tuning.
    """
    if not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
        raise ValueError("invalid soft targets")

    def objective(w):
        z = x @ w
        loss = np.mean(np.logaddexp(0, z) - target * z) + np.dot(w, w) / (
            2 * c * len(x)
        )
        grad = x.T @ (expit(z) - target) / len(x) + w / (c * len(x))
        return loss, grad

    fit = minimize(
        objective,
        np.zeros(x.shape[1]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    if not fit.success:
        raise ValueError(f"student optimizer failed: {fit.message}")
    return fit.x, {
        "iterations": int(fit.nit),
        "objective": float(fit.fun),
        "converged": bool(fit.success),
        "message": str(fit.message),
        "gradient_max_abs": float(np.max(abs(fit.jac))),
    }


def basic(y, p):
    if not len(y):
        return {"n": 0}
    y, p = np.asarray(y), np.asarray(p)
    return {
        "n": len(y),
        "mean_p": float(p.mean()),
        "observed_rate": float(y.mean()),
        "calibration_gap": float(p.mean() - y.mean()),
        "log_loss": float(binary_log_loss_vector(y, p).mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }


def paired(frame, p, ref):
    y = frame.y_true.to_numpy(float)
    if pd.to_datetime(frame.date).dt.to_period("M").nunique() < 2:
        return {"n": len(frame), "unavailable": "fewer than two months"}
    return {
        "n": len(frame),
        "log_loss": monthly_bootstrap(
            binary_log_loss_vector(y, p) - binary_log_loss_vector(y, ref), frame.date
        ),
        "brier": monthly_bootstrap((p - y) ** 2 - (ref - y) ** 2, frame.date),
    }


def diagnostics(frame, column):
    good = frame[column].notna()
    sub = frame.loc[good].reset_index(drop=True)
    y, p = sub.y_true.to_numpy(float), sub[column].to_numpy(float)
    market = sub.market__selected.to_numpy(float)
    slices = {
        "predicted_035_045": (p >= 0.35) & (p < 0.45),
        "market_direction_disagreement": np.isfinite(market)
        & ((p > 0.5) != (market > 0.5)),
        "absolute_market_gap_ge_010": np.isfinite(market) & (abs(p - market) >= 0.1),
        "team_a_market_underdog": market < 0.5,
        "team_a_market_underdog_035_045_prediction": (market < 0.5)
        & (p >= 0.35)
        & (p < 0.45),
        "student_favors_market_underdog": np.isfinite(market)
        & ((p - 0.5) * (market - 0.5) < 0),
        "missing_market": ~np.isfinite(market),
    }
    sliced = {}
    for name, mask in slices.items():
        sliced[name] = basic(y[mask], p[mask])
        if mask.any():
            sliced[name]["paired_baseline79"] = paired(
                sub.loc[mask], p[mask], sub.loc[mask, "p__baseline79"].to_numpy()
            )
    bins = []
    bin_index = np.minimum((p * 10).astype(int), 9)
    for index in range(10):
        mask = bin_index == index
        bins.append(
            {"lower": index / 10, "upper": (index + 1) / 10, **basic(y[mask], p[mask])}
        )
    return {
        "aggregate": probability_metrics(y, p),
        "per_year": {
            str(year): probability_metrics(g.y_true, g[column])
            for year, g in sub.groupby(pd.to_datetime(sub.date).dt.year)
        },
        "paired_baseline79": paired(sub, p, sub.p__baseline79.to_numpy()),
        "slices": sliced,
        "reliability10": bins,
    }


def execute(args, log):
    started = time.monotonic()
    sources = [Path(__file__)] + [
        ROOT / path
        for path in (
            "scripts/benchmark_model_redesign.py",
            "scripts/benchmark_siamese_architectures.py",
            "scripts/train_and_tune_siamese_series.py",
            "scripts/build_siamese_research_dataset.py",
            "src/analysis/probability_metrics.py",
            "src/analysis/metrics.py",
            "src/models/symmetric_series.py",
            "src/models/competition_tiers.py",
            "src/models/redesign_series.py",
            "src/models/correlated_series.py",
            "src/utils/golgg_schema.py",
        )
    ]
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    for path in sources:
        archived = args.output_dir / "source_snapshot" / path.relative_to(ROOT)
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(path.read_bytes())
        if sha256(archived) != code_hashes[str(path.relative_to(ROOT))]:
            raise ValueError("source changed while archiving")
    identity = {"golgg_match_id": "string", "team1_id": "string", "team2_id": "string"}
    legacy_path = ROOT / "data/artifacts/siamese-integrity-v1/snapshots.csv"
    replay_path = ROOT / "data/artifacts/model-redesign-v1/replay/snapshots.csv"
    baseline_path = (
        ROOT
        / "data/artifacts/model-redesign-v1/benchmark-semester/legacy79/predictions.csv"
    )
    odds_path = ROOT / "data/odds.csv"
    inputs = [legacy_path, replay_path, baseline_path, odds_path]
    legacy = (
        pd.read_csv(legacy_path, dtype=identity)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    replay = pd.read_csv(replay_path, dtype=identity)
    replay_audit_path = replay_path.with_name("audit.json")
    if (
        sha256(replay_path)
        != json.loads(replay_audit_path.read_text())["snapshots_sha256"]
    ):
        raise ValueError("replay hash mismatch")
    full, join_audit = align_legacy_context(legacy, replay)
    full = full.sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    odds = pd.read_csv(odds_path, dtype={"golgg_match_id": "string"})
    full, market_join = attach_market(full, odds)
    markets, devig_audit = devig(full.odds_a, full.odds_b)
    reversed_markets, _ = devig(full.odds_b, full.odds_a)
    devig_audit["symmetry_max_error"] = {}
    for name in METHODS:
        good = np.isfinite(markets[name])
        error = float(
            np.max(abs(markets[name][good] + reversed_markets[name][good] - 1))
        )
        if error > 1e-6:
            raise ValueError(f"market symmetry failed: {name}")
        devig_audit["symmetry_max_error"][name] = error
    source_devig, source_audit = devig(
        pd.to_numeric(odds.avg_odds_home, errors="coerce"),
        pd.to_numeric(odds.avg_odds_away, errors="coerce"),
    )
    odds_audit = {
        "rows": len(odds),
        "fields": list(odds),
        "field_nonnull": {c: int(odds[c].notna().sum()) for c in odds},
        "duplicate_id_rows": int(odds.golgg_match_id.duplicated(keep=False).sum()),
        "odds_date_bounds": [str(odds.odds_date.min()), str(odds.odds_date.max())],
        "source_prices": source_audit,
        "joined_prices": devig_audit,
        "join": market_join,
        "primary_limitation": "avg_odds_home/away are aggregated closing prices without trustworthy quote timestamps or verified same-bookmaker/same-time executable pairs; odds_date and golgg_date are date fields, not quote timestamps",
    }
    (args.output_dir / "odds_audit.json").write_text(
        json.dumps(odds_audit, indent=2, allow_nan=False) + "\n"
    )
    x, names = build_training_features(full)
    if not np.isfinite(x).all() or x.shape[1] != 79:
        raise ValueError("canonical79 feature contract failed")
    y = full.y_true.to_numpy(float)
    eligible = np.logical_and.reduce([np.isfinite(markets[n]) for n in METHODS])
    sums = 1 / full.odds_a.to_numpy() + 1 / full.odds_b.to_numpy()
    eligible &= sums >= 1
    baseline = (
        pd.read_csv(baseline_path, dtype=identity, low_memory=False)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    blocks, folds = [], []
    for year in (2024, 2025, 2026):
        masks = temporal_blocks(full.date, year, protocol="semester-calibration")
        train, cal, test = (masks[n] for n in ("train", "calibration", "test"))
        covered = train & eligible
        if any(not m.any() for m in masks.values()) or not covered.any():
            raise ValueError(f"empty split: {year}")
        select = masks["select"] & eligible
        selection = {name: basic(y[select], markets[name][select]) for name in METHODS}
        if not select.any():
            raise ValueError(f"no prior select quotes for {year}")
        selected = min(METHODS, key=lambda name: selection[name]["log_loss"])
        fold = {
            "year": year,
            "selected_devig": selected,
            "select_devig_metrics_common_rows": selection,
            "stages": {
                n: {
                    "n": int(m.sum()),
                    "min": str(full.loc[m, "date"].min()),
                    "max": str(full.loc[m, "date"].max()),
                    "market_covered_n": int((m & eligible).sum()),
                    "market_coverage": float(eligible[m].mean()),
                }
                for n, m in masks.items()
            },
            "models": {},
        }
        part = full.loc[
            test,
            [
                "golgg_match_id",
                "date",
                "y_true",
                "best_of",
                "competition_tier",
                "odds_a",
                "odds_b",
            ],
        ].copy()
        part["market__selected"] = markets[selected][test]
        part["selected_devig"] = selected
        for name in METHODS:
            part["market__" + name] = markets[name][test]
        # Matched controls share scaler, train rows, regularization, calibration rows.
        matched_scaler = StandardScaler(with_mean=False).fit(x[covered])
        full_scaler = StandardScaler(with_mean=False).fit(x[train])
        variants = [
            ("md_hard_full", train, full_scaler, 0.0, None),
            ("md_hard_matched", covered, matched_scaler, 0.0, None),
        ]
        variants += [
            (f"md_{method}_w{int(w * 100):02d}", covered, matched_scaler, w, method)
            for method in METHODS
            for w in WEIGHTS
        ]
        for name, training, scaler, weight, method in variants:
            target = (
                y[training]
                if method is None
                else (1 - weight) * y[training] + weight * markets[method][training]
            )
            coefficients, optimization = train_student(
                scaler.transform(x[training]), target
            )
            zcal = scaler.transform(x[cal]) @ coefficients
            slope = fit_platt_scaling(zcal, y[cal])
            ztest = scaler.transform(x[test]) @ coefficients
            raw = expit(ztest)
            p = expit(slope * ztest)
            if np.any((p <= 0) | (p >= 1)) or np.any((raw <= 0) | (raw >= 1)):
                raise ValueError("student probabilities reached a boundary")
            reverse = expit(slope * (scaler.transform(-x[test]) @ coefficients))
            symmetry = float(np.max(abs(p + reverse - 1)))
            if symmetry > 1e-6:
                raise ValueError("student symmetry failed")
            part["p__" + name], part["raw__" + name] = p, raw
            payload = {
                "coefficients": coefficients,
                "scaler": scaler,
                "calibration_slope": slope,
                "feature_names": names,
                "year": year,
                "variant": name,
                "teacher_weight": weight,
                "teacher_method": method,
                "inference": "expit(calibration_slope * (scaler.transform(canonical79) @ coefficients)); no odds",
            }
            model_path = args.output_dir / f"{year}-{name}.joblib"
            joblib.dump(payload, model_path)
            loaded = joblib.load(model_path)
            np.testing.assert_array_equal(
                expit(
                    loaded["calibration_slope"]
                    * (loaded["scaler"].transform(x[test]) @ loaded["coefficients"])
                ),
                p,
            )
            stop_z = scaler.transform(x[masks["stop"]]) @ coefficients
            fold["models"][name] = {
                "training_n": int(training.sum()),
                "teacher_weight": weight,
                "teacher_method": method,
                "calibration_slope": slope,
                "symmetry_max_error": symmetry,
                "optimization": optimization,
                "stop_raw_metrics_diagnostic_only": basic(
                    y[masks["stop"]], expit(stop_z)
                ),
                "artifact": model_path.name,
                "sha256": sha256(model_path),
            }
            log(
                {
                    "year": year,
                    "variant": name,
                    "status": "completed",
                    **basic(y[test], p),
                }
            )
        for weight in WEIGHTS:
            name = f"md_selected_w{int(weight * 100):02d}"
            original = f"md_{selected}_w{int(weight * 100):02d}"
            part["p__" + name] = part["p__" + original]
            part["raw__" + name] = part["raw__" + original]
        blocks.append(part)
        folds.append(fold)
    result = (
        pd.concat(blocks).sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    )
    if result.golgg_match_id.duplicated().any():
        raise ValueError("duplicate test IDs")
    for col in ("golgg_match_id", "date", "y_true", "best_of"):
        np.testing.assert_array_equal(result[col].to_numpy(), baseline[col].to_numpy())
    result["p__baseline79"] = baseline.fixed_ridge.to_numpy()
    result.to_csv(args.output_dir / "predictions.csv", index=False)
    # Direct market probabilities are intentionally not recalibrated or imputed.
    # Use a separate file so every p__ column in main predictions is full-coverage.
    market_frame = result.loc[result.market__multiplicative.notna()].copy()
    for method in METHODS:
        market_frame["p__market_" + method] = market_frame["market__" + method]
    market_frame.to_csv(args.output_dir / "market_predictions.csv", index=False)
    sport_columns = [c for c in result if c.startswith("p__")]
    metrics = {c[3:]: diagnostics(result, c) for c in sport_columns}
    market_metrics = {
        method: diagnostics(result, "market__" + method) for method in METHODS
    }
    comparisons = {}
    for c in sport_columns:
        if c.startswith("p__md_"):
            comparisons[c[3:]] = paired(
                result, result[c].to_numpy(), result.p__md_hard_matched.to_numpy()
            )
    if code_hashes != {str(path.relative_to(ROOT)): sha256(path) for path in sources}:
        raise ValueError("source changed during experiment")
    summary = {
        "experiment": "EXP-087",
        "status": "completed",
        "exploratory_retrospective_only": True,
        "variants": [c[3:] for c in sport_columns],
        "market_variants": list(METHODS),
        "protocol": "semester-calibration",
        "folds": folds,
        "join_audit": join_audit,
        "coverage": {
            "common_test_n": len(result),
            "baseline_test_n": len(baseline),
            "eligibility_loss": 0,
            "market_test_n": len(market_frame),
        },
        "metrics": metrics,
        "market_metrics": market_metrics,
        "paired_matched_control": comparisons,
        "training": {
            "seed": 87,
            "bootstrap_seed": 82,
            "bootstrap_month_resamples": 5000,
            "C": 0.1,
            "loss": "mean direct Bernoulli BCE(soft target, sigmoid(Xw)) + ||w||^2/(2*C*n)",
            "teacher_weights": list(WEIGHTS),
            "solver": "L-BFGS-B",
            "maxiter": 3000,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "intercept": False,
            "scale_centering": False,
            "stopping": "numerical convergence on TRAIN objective; STOP is diagnostic only, no checkpoint hyperselection",
            "devig_selection": "lowest raw market log loss on prior SELECT common overround rows; fixed weights not selected",
            "calibration": "independent positive zero-intercept slope, all H2 prior-year hard outcomes only",
            "feature_names": names,
        },
        "baseline_source": str(baseline_path.relative_to(ROOT))
        + " fixed_ridge diagnostic column; full historical sports-data fit",
        "input_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in inputs + [replay_audit_path]
        },
        "code_sha256": code_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                n: version(n)
                for n in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]
            },
            "threads": 1,
        },
        "limitations": [
            odds_audit["primary_limitation"],
            "Historical legacy rating provenance and stored EXP081 training cutoff UNVERIFIED",
            "Previously inspected holdout: no superiority, promotion, or live-profit inference",
            "Closing-price teacher is retrospective market-informed supervision, not evidence of executable edge",
            "Canonical79 has inherited provenance limitations; this run adds no target draft/patch/roster/outcome or same-day update",
            "Market teacher uses TRAIN rows only, never calibration/test market labels; coverage matched to hard control",
            "Linear student does not test nonlinear recovery of market knowledge",
        ],
        "sources": [SOURCE],
        "runtime_seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    artifact_paths = sorted(args.output_dir.glob("*.joblib")) + [
        args.output_dir / "predictions.csv",
        args.output_dir / "market_predictions.csv",
        args.output_dir / "summary.json",
        args.output_dir / "odds_audit.json",
    ]
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {p.name: sha256(p) for p in artifact_paths},
                "command": sys.argv,
            },
            indent=2,
        )
        + "\n"
    )
    log(
        {
            "status": "completed",
            "n": len(result),
            "runtime_seconds": summary["runtime_seconds"],
        }
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "metrics": {n: m["aggregate"] for n, m in metrics.items()},
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.random.seed(87)

    def log(event):
        with (args.output_dir / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event), flush=True)

    log({"status": "started", "command": sys.argv})
    try:
        execute(args, log)
    except Exception:
        failure = {
            "status": "failed",
            "traceback": traceback.format_exc(),
            "command": sys.argv,
            "code_sha256": sha256(Path(__file__)),
        }
        (args.output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2) + "\n"
        )
        log({"status": "failed", "failure": "failure.json"})
        raise


if __name__ == "__main__":
    main()
