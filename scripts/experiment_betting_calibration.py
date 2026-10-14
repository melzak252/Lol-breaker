#!/usr/bin/env python3
"""EXP-088: retrospective calibration under a fixed odds-based selection policy.

No placement, wallet/ROI simulation, database access, or production changes.
Historical averaged closing prices have no trustworthy quote timestamps: every
selection below is a DIAGNOSTIC, not a reconstruction of executable bets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_exp081_failures import bootstrap
from scripts.benchmark_model_redesign import align_legacy_context, temporal_blocks
from scripts.benchmark_siamese_architectures import attach_market, probability_metrics
from scripts.build_siamese_research_dataset import sha256
from scripts.train_and_tune_siamese_series import (
    build_training_features,
    fit_platt_scaling,
)
from src.analysis.probability_metrics import binary_log_loss_vector
from src.models.redesign_series import SeriesEstimator

EPS = np.finfo(float).eps
TAX = 0.12
MIN_EV = 0.05
METADATA = ["golgg_match_id", "date", "y_true", "best_of", "competition_tier"]
IDENTITIES = {"golgg_match_id": "string", "team1_id": "string", "team2_id": "string"}


def open_probability(p):
    return np.clip(p, EPS, 1 - EPS)


def policy(p, odds_a, odds_b):
    """At most one side per series; tie/no valid paired quote means no selection."""
    p, oa, ob = (np.asarray(x, float) for x in (p, odds_a, odds_b))
    if (
        p.shape != oa.shape
        or p.shape != ob.shape
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("unaligned or invalid policy probabilities")
    eligible = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)
    ev_a = p * oa * (1 - TAX) - 1
    ev_b = (1 - p) * ob * (1 - TAX) - 1
    side_a = ev_a > ev_b
    selected = eligible & (np.maximum(ev_a, ev_b) > MIN_EV) & (ev_a != ev_b)
    return selected, side_a


def calibration_context(frame):
    """Only predeclared even format/tier context, not posthoc selected league IDs."""
    labels = [
        "intercept",
        "bo3",
        "bo5",
        "major",
        "international",
        "development",
        "minor_top_level",
    ]
    return np.column_stack(
        (
            np.ones(len(frame)),
            frame.best_of == 3,
            frame.best_of == 5,
            frame.competition_tier == "major",
            frame.competition_tier == "international",
            frame.competition_tier == "development",
            frame.competition_tier == "minor_top_level",
        )
    ).astype(float), labels


def fit_temperatures(z, y, context, prior_slope):
    """Positive subgroup slopes; fixed L2 penalty, no search on test outcomes."""
    prior = np.log(prior_slope)
    penalty = 0.01

    def objective(beta):
        slope = np.exp(prior + context @ beta)
        logits = slope * z
        loss = (
            np.mean(np.logaddexp(0, logits) - y * logits)
            + penalty * np.dot(beta[1:], beta[1:]) / 2
        )
        gradient = context.T @ ((expit(logits) - y) * logits) / len(y)
        gradient[1:] += penalty * beta[1:]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(context.shape[1]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-2, 2)] * context.shape[1],
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"conditional calibration failed: {result.message}")
    return result.x, {
        "prior_slope": prior_slope,
        "l2_penalty": penalty,
        "beta": result.x.tolist(),
        "bounds": [-2, 2],
        "boundary": bool(np.any(np.abs(result.x) > 1.9999)),
        "optimizer": str(result.message),
    }


def fit_weighted_slope(z, y, weights):
    def loss(log_slope):
        logits = np.exp(log_slope) * z
        return float(np.average(np.logaddexp(0, logits) - y * logits, weights=weights))

    result = minimize_scalar(
        loss, bounds=(-4, 4), method="bounded", options={"xatol": 1e-10}
    )
    if not result.success:
        raise RuntimeError("selected-weighted calibration optimizer failed")
    return float(np.exp(result.x)), {
        "log_slope_bounds": [-4, 4],
        "boundary": bool(abs(result.x) > 3.9999),
    }


def reliability(y, p, dates):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not len(y):
        return {"n": 0, "status": "no eligible observations"}
    # Wilson intervals are descriptive IID references; monthly error bootstrap is primary.
    n = len(y)
    observed = float(y.mean())
    a = 1.959963984540054**2
    center = (observed + a / (2 * n)) / (1 + a / n)
    half = (
        1.959963984540054
        * np.sqrt(observed * (1 - observed) / n + a / (4 * n * n))
        / (1 + a / n)
    )
    return {
        "n": n,
        "mean_probability": float(p.mean()),
        "observed_win_fraction": observed,
        "signed_error_probability_minus_result": float(np.mean(p - y)),
        "signed_error_monthly_ci": bootstrap(
            p - y, pd.Series(dates).reset_index(drop=True)
        ),
        "iid_wilson_reference": [center - half, center + half],
        "log_loss": float(binary_log_loss_vector(y, p).mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }


def reliability_groups(frame, p, selected, side_a):
    selected = np.asarray(selected, bool)
    outcomes = np.where(side_a, frame.y_true, 1 - frame.y_true)
    bet_p = np.where(side_a, p, 1 - p)
    results = {
        "all_selected": reliability(
            outcomes[selected], bet_p[selected], frame.date[selected]
        )
    }
    for lo, hi in (
        (0.0, 0.2),
        (0.2, 0.35),
        (0.35, 0.45),
        (0.45, 0.6),
        (0.6, 0.75),
        (0.75, 1.01),
    ):
        mask = selected & (bet_p >= lo) & (bet_p < hi)
        results[f"p_{lo:g}_{hi:g}"] = reliability(
            outcomes[mask], bet_p[mask], frame.date[mask]
        )
    for key in ("best_of", "competition_tier"):
        for value in sorted(frame[key].unique()):
            mask = selected & (frame[key].to_numpy() == value)
            results[f"{key}:{value}"] = reliability(
                outcomes[mask], bet_p[mask], frame.date[mask]
            )
    for year in sorted(frame.date.str[:4].unique()):
        mask = selected & frame.date.str.startswith(year).to_numpy()
        results["year:" + year] = reliability(
            outcomes[mask], bet_p[mask], frame.date[mask]
        )
    return results


def evaluate(frame, prediction_columns):
    baseline = frame["p__baseline79"].to_numpy()
    y = frame.y_true.to_numpy()
    fixed_mask, fixed_side = policy(baseline, frame.odds_a, frame.odds_b)
    common = frame.market.notna().to_numpy()
    results = {}
    selected_rows = []
    for name in prediction_columns:
        p = frame[name].to_numpy()
        mask, side = policy(p, frame.odds_a, frame.odds_b)
        fixed_y = np.where(fixed_side, y, 1 - y)
        fixed_p = np.where(fixed_side, p, 1 - p)
        fixed_base = np.where(fixed_side, baseline, 1 - baseline)
        results[name] = {
            "overall": probability_metrics(y, p),
            "paired_LL_vs_baseline": bootstrap(
                binary_log_loss_vector(y, p) - binary_log_loss_vector(y, baseline),
                frame.date,
            ),
            "paired_Brier_vs_baseline": bootstrap(
                (p - y) ** 2 - (baseline - y) ** 2, frame.date
            ),
            "market_common": probability_metrics(y[common], p[common]),
            "own_selection": reliability_groups(frame, p, mask, side),
            "fixed_baseline_selection": reliability_groups(
                frame, p, fixed_mask, fixed_side
            ),
            "fixed_selection_paired_LL": bootstrap(
                binary_log_loss_vector(fixed_y[fixed_mask], fixed_p[fixed_mask])
                - binary_log_loss_vector(fixed_y[fixed_mask], fixed_base[fixed_mask]),
                frame.date[fixed_mask],
            ),
            "fixed_reference_40": reliability(
                fixed_y[fixed_mask & (fixed_base >= 0.35) & (fixed_base < 0.45)],
                fixed_p[fixed_mask & (fixed_base >= 0.35) & (fixed_base < 0.45)],
                frame.date[fixed_mask & (fixed_base >= 0.35) & (fixed_base < 0.45)],
            ),
            "selection_overlap": {
                "own_n": int(mask.sum()),
                "reference_n": int(fixed_mask.sum()),
                "same_selected_side_n": int(
                    (mask & fixed_mask & (side == fixed_side)).sum()
                ),
            },
        }
        rows = frame.loc[mask, METADATA].copy()
        rows["model"] = name
        rows["side_a"] = side[mask]
        rows["selected_probability"] = np.where(side, p, 1 - p)[mask]
        rows["selected_outcome"] = np.where(side, y, 1 - y)[mask]
        rows["decimal_odds"] = np.where(side, frame.odds_a, frame.odds_b)[mask]
        selected_rows.append(rows)
    return results, pd.concat(selected_rows, ignore_index=True)


def run_calibration(
    full,
    reference,
    benchmark_dir,
    output,
    *,
    years=(2024, 2025, 2026),
    train_start="2020-01-01",
):
    x79, names = build_training_features(full)
    x = np.column_stack(
        (x79, (full.best_of == 3).astype(float), (full.best_of == 5).astype(float))
    )
    expected_names = tuple(["d_" + n for n in names] + ["c_bo3", "c_bo5"])
    c, context_names = calibration_context(full)
    rows, metadata = [], []
    for year in years:
        masks = temporal_blocks(
            full.date, year, protocol="semester-calibration", train_start=train_start
        )
        cal, select, test = (masks[k] for k in ("calibration", "select", "test"))
        artifact = benchmark_dir / "legacy79" / f"{year}-fixed_ridge.joblib"
        model = SeriesEstimator.load(artifact)
        if model.feature_names != expected_names:
            raise ValueError("baseline artifact feature contract differs")
        raw = model.raw_probability(x)
        z = logit(raw)
        y = full.y_true.to_numpy()
        slope = fit_platt_scaling(z[cal], y[cal])
        np.testing.assert_allclose(slope, model.slope, atol=1e-10, rtol=0)
        part = full.loc[test, METADATA].copy()
        part["p__baseline79"] = model.predict(x[test])
        part["raw__baseline79"] = raw[test]
        beta, temperature_meta = fit_temperatures(z[cal], y[cal], c[cal], slope)
        subgroup_slope = np.exp(np.log(slope) + c[test] @ beta)
        part["p__format_tier_temperature"] = open_probability(
            expit(subgroup_slope * z[test])
        )
        # The preliminary selector's calibrator is frozen on Q2, before H2 calibration.
        probe_slope = fit_platt_scaling(z[select], y[select])
        probe = open_probability(expit(probe_slope * z[cal]))
        selected, _ = policy(probe, full.loc[cal, "odds_a"], full.loc[cal, "odds_b"])
        weights = 1 + 4 * selected.astype(float)
        weighted_slope, weighted_meta = fit_weighted_slope(z[cal], y[cal], weights)
        part["p__selected_weighted_temperature"] = open_probability(
            expit(weighted_slope * z[test])
        )
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
            np.r_[raw[cal], 1 - raw[cal]], np.r_[y[cal], 1 - y[cal]]
        )

        def isotonic(p):
            return open_probability((iso.predict(p) + 1 - iso.predict(1 - p)) / 2)

        part["p__symmetric_isotonic"] = isotonic(raw[test])
        # Actual reverse evaluations, not an assertion against fabricated 1-p values.
        reverse_raw = model.raw_probability(np.column_stack((-x79[test], x[test, -2:])))
        reverse_z = logit(reverse_raw)
        reverse_predictions = {
            "p__baseline79": open_probability(expit(slope * reverse_z)),
            "p__format_tier_temperature": open_probability(
                expit(subgroup_slope * reverse_z)
            ),
            "p__selected_weighted_temperature": open_probability(
                expit(weighted_slope * reverse_z)
            ),
            "p__symmetric_isotonic": isotonic(reverse_raw),
        }
        for name, reverse in reverse_predictions.items():
            np.testing.assert_allclose(
                part[name].to_numpy() + reverse, 1, atol=1e-12, rtol=0
            )
            chosen, chosen_side = policy(
                part[name], full.loc[test, "odds_a"], full.loc[test, "odds_b"]
            )
            reverse_chosen, reverse_side = policy(
                reverse, full.loc[test, "odds_b"], full.loc[test, "odds_a"]
            )
            np.testing.assert_array_equal(chosen, reverse_chosen)
            np.testing.assert_array_equal(chosen_side[chosen], ~reverse_side[chosen])
        rows.append(part)
        record = {
            "year": year,
            "base_artifact": str(artifact),
            "base_sha256": sha256(artifact),
            "context_names": context_names,
            "format_tier_temperature": temperature_meta,
            "selected_weighted_temperature": {
                "probe_slope": probe_slope,
                "final_slope": weighted_slope,
                "selected_calibration_n": int(selected.sum()),
                "weights_unselected_selected": [1, 5],
                **weighted_meta,
            },
            "isotonic": {
                "x_thresholds": iso.X_thresholds_.tolist(),
                "y_thresholds": iso.y_thresholds_.tolist(),
                "paired_mirror_fit": True,
            },
            "stages": {
                stage: {
                    "n": int(mask.sum()),
                    "min_date": full.loc[mask, "date"].min(),
                    "max_date": full.loc[mask, "date"].max(),
                }
                for stage, mask in masks.items()
            },
        }
        metadata.append(record)
        (output / f"{year}-calibrators.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
    predictions = pd.concat(rows, ignore_index=True)
    if predictions.golgg_match_id.tolist() != reference.golgg_match_id.tolist():
        raise ValueError("calibration experiment changes reference cohort")
    np.testing.assert_allclose(
        predictions.p__baseline79, reference.fixed_ridge, atol=1e-12, rtol=0
    )
    return predictions, metadata


def add_external(frame, paths):
    audits = []
    for path in paths:
        external = pd.read_csv(path, dtype=IDENTITIES, low_memory=False)
        if external.golgg_match_id.duplicated().any():
            raise ValueError(f"duplicate external prediction IDs: {path}")
        if set(external.golgg_match_id) != set(frame.golgg_match_id):
            raise ValueError(
                f"external cohort differs: {path}; requires explicit matched-cohort experiment"
            )
        external = (
            external.set_index("golgg_match_id").loc[frame.golgg_match_id].reset_index()
        )
        for key in METADATA:
            if not np.array_equal(external[key].to_numpy(), frame[key].to_numpy()):
                raise ValueError(f"external {key} alignment mismatch: {path}")
        columns = [c for c in external if c.startswith("p__") and c != "p__baseline79"]
        if not columns:
            raise ValueError(f"no model predictions: {path}")
        prefix = path.parent.parent.name
        for name in columns:
            target = f"p__{prefix}__{name[3:]}"
            if target in frame:
                raise ValueError(f"duplicate model name: {target}")
            values = external[name].to_numpy(float)
            if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
                raise ValueError(f"invalid external probabilities: {name}")
            frame[target] = values
        audits.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "rows": len(external),
                "models": columns,
            }
        )
    return frame, audits


def market_calibration_followup(
    full,
    reference,
    benchmark_dir,
    output,
    *,
    years=(2024, 2025, 2026),
    registered_sports_column="stored_exp081",
    train_start="2020-01-01",
):
    """Explicit follow-up after run01: market-conditioned, NOT sports-only."""
    from betting_app.core.models.engine import PredictionEngine
    from betting_app.core.models.registry import get_active_hybrid

    hybrid_spec = get_active_hybrid()
    x79, _ = build_training_features(full)
    x = np.column_stack(
        (x79, (full.best_of == 3).astype(float), (full.best_of == 5).astype(float))
    )
    available = full.market.notna().to_numpy()
    rows, records = [], []
    for year in years:
        masks = temporal_blocks(
            full.date, year, protocol="semester-calibration", train_start=train_start
        )
        cal = masks["calibration"] & available
        test = masks["test"] & available
        base = SeriesEstimator.load(
            benchmark_dir / "legacy79" / f"{year}-fixed_ridge.joblib"
        )
        z = logit(base.raw_probability(x))
        y = full.y_true.to_numpy()
        xcal = np.column_stack((z[cal], logit(full.loc[cal, "market"])))
        xtest = np.column_stack((z[test], logit(full.loc[test, "market"])))
        part = full.loc[test, METADATA + ["odds_a", "odds_b", "market"]].copy()
        part["p__baseline79"] = base.predict(x[test])
        part["p__market_multiplicative"] = full.loc[test, "market"]
        part["p__fixed_half_logit_hybrid"] = open_probability(
            expit(0.5 * (base.slope * xtest[:, 0] + xtest[:, 1]))
        )
        probe_slope = fit_platt_scaling(z[masks["select"]], y[masks["select"]])
        selected, _ = policy(
            expit(probe_slope * z[cal]),
            full.loc[cal, "odds_a"],
            full.loc[cal, "odds_b"],
        )
        for weighted in (False, True):
            weight = 1 + 4 * selected.astype(float) if weighted else np.ones(cal.sum())
            penalty = 0.01

            def objective(coef):
                logits = xcal @ coef
                residual = expit(logits) - y[cal]
                loss = (
                    np.average(
                        np.logaddexp(0, logits) - y[cal] * logits, weights=weight
                    )
                    + penalty * np.sum((coef - 0.5) ** 2) / 2
                )
                gradient = xcal.T @ (weight * residual) / weight.sum() + penalty * (
                    coef - 0.5
                )
                return loss, gradient

            fit = minimize(
                objective,
                np.array([0.5, 0.5]),
                jac=True,
                method="L-BFGS-B",
                bounds=[(0, 4), (0, 4)],
                options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 2000},
            )
            if not fit.success:
                raise RuntimeError(
                    f"market-conditioned calibration failed: {fit.message}"
                )
            name = "p__market_conditioned" + ("_selected_weighted" if weighted else "")
            part[name] = open_probability(expit(xtest @ fit.x))
            reverse = open_probability(expit((-xtest) @ fit.x))
            np.testing.assert_allclose(part[name] + reverse, 1, atol=1e-12, rtol=0)
            records.append(
                {
                    "year": year,
                    "variant": name,
                    "calibration_n": int(cal.sum()),
                    "test_n": int(test.sum()),
                    "coefficient_sports_market": fit.x.tolist(),
                    "penalty": penalty,
                    "prior_coefficient": [0.5, 0.5],
                    "coefficient_bounds": [0, 4],
                    "weights": [1, 5] if weighted else [1, 1],
                    "boundary": bool(np.any(fit.x < 1e-6) | np.any(fit.x > 3.999999)),
                }
            )
        rows.append(part)
    frame = pd.concat(rows, ignore_index=True)
    source = reference.set_index("golgg_match_id").loc[frame.golgg_match_id]
    np.testing.assert_allclose(
        frame.p__baseline79, source.fixed_ridge, atol=1e-12, rtol=0
    )
    replay_column = f"p__registered_blend_{registered_sports_column}_diagnostic"
    frame[replay_column] = [
        PredictionEngine.blend_with_market(float(p), float(m), hybrid_spec)
        for p, m in zip(source[registered_sports_column], frame.market)
    ]
    reverse = [
        PredictionEngine.blend_with_market(float(1 - p), float(1 - m), hybrid_spec)
        for p, m in zip(source[registered_sports_column], frame.market)
    ]
    np.testing.assert_allclose(
        frame[replay_column] + reverse,
        1,
        atol=1e-12,
        rtol=0,
    )
    results, selections = evaluate(frame, [n for n in frame if n.startswith("p__")])
    frame.to_csv(output / "market_conditioned_predictions.csv", index=False)
    selections.to_csv(output / "market_conditioned_selections.csv", index=False)
    plot_reliability(results, output, prefix="market_conditioned")
    (output / "market_calibrators.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n"
    )
    return {
        "selected_after_run01": True,
        "market_required_at_inference": True,
        "sports_only": False,
        "rows": len(frame),
        "models": results,
        "calibrators": records,
        "registered_blend_replay": {
            "alpha": hybrid_spec.alpha,
            "temperature": hybrid_spec.temperature,
            "blending_mode": hybrid_spec.blending_mode,
            "sports_predictions": registered_sports_column,
            "source_provenance": (
                "See reference fold metadata; stored EXP081 training cutoff remains "
                "unverified. Registry rule applied to fixed_ridge is not an "
                "operational EXP081 prediction replay."
            ),
            "odds": "historical closing averages; NOT recorded live recommendations",
        },
    }


def plot_reliability(results, output, prefix="selected"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    chosen = [
        "p__baseline79",
        "p__format_tier_temperature",
        "p__selected_weighted_temperature",
        "p__symmetric_isotonic",
    ]
    if prefix != "selected":
        chosen = list(results)
    bins = [
        "p_0_0.2",
        "p_0.2_0.35",
        "p_0.35_0.45",
        "p_0.45_0.6",
        "p_0.6_0.75",
        "p_0.75_1.01",
    ]
    for axis, selection in zip(axes, ("own_selection", "fixed_baseline_selection")):
        for name in chosen:
            rows = [results[name][selection][key] for key in bins]
            rows = [r for r in rows if r["n"] >= 20]
            axis.plot(
                [r["mean_probability"] for r in rows],
                [r["observed_win_fraction"] for r in rows],
                "o-",
                label=name.removeprefix("p__"),
            )
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set(
            xlabel="Mean selected-side probability",
            ylabel="Observed selected-side win fraction",
            title=selection.replace("_", " ") + " (bins N≥20)",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.legend(fontsize=7)
    fig.suptitle(
        "EXP-088: retrospective closing-price selection, tax 12%, EV>5%; NOT executable bets"
    )
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_calibration.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=ROOT / "data/artifacts/model-redesign-v1/benchmark-semester",
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=ROOT / "data/artifacts/siamese-integrity-v1/snapshots.csv",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=ROOT / "data/artifacts/model-redesign-v1/replay/snapshots.csv",
    )
    parser.add_argument("--odds", type=Path, default=ROOT / "data/odds.csv")
    parser.add_argument("--additional-predictions", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--include-market-calibration",
        action="store_true",
        help="Explicit exploratory follow-up after run01, not a sports-only model",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (args.output_dir / "runner_source.py").write_text(Path(__file__).read_text())
    read_frame = lambda path: (
        pd.read_csv(path, dtype=IDENTITIES, low_memory=False)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    legacy, replay = read_frame(args.legacy), read_frame(args.replay)
    full, join_audit = align_legacy_context(legacy, replay)
    odds = pd.read_csv(args.odds, dtype={"golgg_match_id": "string"}, low_memory=False)
    full, market_audit = attach_market(full, odds)
    reference_path = args.benchmark_dir / "legacy79/predictions.csv"
    reference = read_frame(reference_path)
    current_market = full.set_index("golgg_match_id").loc[reference.golgg_match_id]
    for column in ("odds_a", "odds_b", "market"):
        np.testing.assert_allclose(
            current_market[column],
            reference[column],
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        )
    predictions, calibrators = run_calibration(
        full, reference, args.benchmark_dir, args.output_dir
    )
    aligned = predictions.merge(
        reference[["golgg_match_id", "odds_a", "odds_b", "market", "stored_exp081"]],
        on="golgg_match_id",
        validate="one_to_one",
        sort=False,
    )
    aligned["p__stored_exp081_diagnostic"] = aligned.stored_exp081
    aligned, external = add_external(aligned, args.additional_predictions)
    names = [name for name in aligned if name.startswith("p__")]
    results, selections = evaluate(aligned, names)
    predictions = aligned[
        METADATA + [c for c in aligned if c.startswith(("p__", "raw__"))]
    ]
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    selections.to_csv(args.output_dir / "selected_diagnostic_rows.csv", index=False)
    plot_reliability(results, args.output_dir)
    source_paths = [
        Path(__file__),
        args.legacy,
        args.replay,
        args.odds,
        reference_path,
        ROOT / "scripts/analyze_exp081_failures.py",
        ROOT / "scripts/benchmark_model_redesign.py",
        ROOT / "scripts/benchmark_siamese_architectures.py",
        ROOT / "scripts/train_and_tune_siamese_series.py",
        ROOT / "src/models/redesign_series.py",
    ]
    source_paths.extend(
        ROOT / path
        for path in (
            "betting_app/core/models/engine.py",
            "betting_app/core/models/registry.py",
            "betting_app/core/models/contract.py",
        )
    )
    source_paths.extend(
        ROOT / path
        for path in (
            "scripts/build_siamese_research_dataset.py",
            "src/analysis/probability_metrics.py",
            "src/analysis/metrics.py",
            "src/models/symmetric_series.py",
            "src/models/competition_tiers.py",
            "src/models/siamese_series.py",
            "src/models/correlated_series.py",
            "src/utils/golgg_schema.py",
        )
    )
    for path in source_paths:
        if path.suffix == ".py":
            archived = (
                args.output_dir / "source_snapshot" / path.resolve().relative_to(ROOT)
            )
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(path.read_bytes())
    summary = {
        "experiment": "EXP-088",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "retrospective conditional calibration; no live qualification",
        "command": sys.argv,
        "environment": {
            "python": sys.version,
            "packages": {
                name: version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
            },
        },
        "rows": len(aligned),
        "market_rows": int(aligned.market.notna().sum()),
        "policy": {
            "tax": TAX,
            "min_ev_strictly_greater": MIN_EV,
            "one_side_per_series": True,
            "tie": "no selection",
            "fixed_before_run": True,
            "executable_rows_with_complete_timestamp_chain": 0,
        },
        "results": results,
        "calibrators": calibrators,
        "external_predictions": external,
        "join_audit": join_audit,
        "market_audit": market_audit,
        "runtime_seconds": time.monotonic() - started,
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "limitations": [
            "Historical odds are averaged closing prices, not independently timestamped paired quotes.",
            "No executable betting returns or profit claim.",
            "Legacy features and stored EXP081 training cutoff are not certified.",
            "Inspected historical holdout is exploratory; no promotion based on these results.",
            "Selected models may change selected cohorts; fixed baseline mask and own masks are both reported.",
            "Subgroup/bin comparisons have no multiple-comparison correction.",
            "Near-40% bin is [35%,45%), not a claim of pointwise or individual-match probability truth.",
            "Monthly error bootstrap is primary; Wilson intervals assume independent trials and are descriptive only.",
        ],
    }
    if args.include_market_calibration:
        summary["market_conditioned_followup"] = market_calibration_followup(
            full, reference, args.benchmark_dir, args.output_dir
        )
    summary["runtime_seconds"] = time.monotonic() - started
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "market_rows": summary["market_rows"],
                "models": {
                    name: {
                        "LL": r["overall"]["log_loss"],
                        "selected": r["own_selection"]["all_selected"],
                        "near40": r["own_selection"]["p_0.35_0.45"],
                    }
                    for name, r in results.items()
                },
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
