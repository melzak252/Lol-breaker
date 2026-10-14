#!/usr/bin/env python3
"""Exploratory EXP081 Benter-style decision calibration; never a live backtest.

Run with .venv/bin/python and a new --output-dir. Sports predictions and frozen
models are read-only. Close is primary; open is a separate timing sensitivity.
Neither historical replay nor these quote phases is certified point-in-time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_siamese_architectures import (
    attach_market,
    monthly_bootstrap,
    probability_metrics,
)
from scripts.corrected_history_evaluation import frozen_hashes, sha256
from scripts.experiment_betting_calibration import MIN_EV, TAX, policy, reliability
from src.analysis.probability_metrics import binary_log_loss_vector

PREDICTION_COLUMN = "recipe__reported_regularization_bagging"
L2 = 0.001
SEED = 82
RESAMPLES = 5000
EPS = 2.0 ** -40  # Exactly representable complementary endpoints preserve symmetry.
YEARS = (2025, 2026)
PHASES = ("close", "open")
MODELS = ("raw_sports", "sports_calibrated", "raw_market", "market_calibrated", "benter")
FEATURES = {
    "sports_calibrated": ("sports",),
    "market_calibrated": ("market",),
    "benter": ("sports", "market"),
}
METADATA = ("golgg_match_id", "team1_id", "team2_id", "date", "best_of", "y_true")
LIMITATIONS = [
    "Historical replay and repaired quotes lack certified point-in-time provenance.",
    "STS close is a retrospective diagnostic; STS open is a separate sensitivity, not certified decision-time data.",
    "All evaluation history was inspected previously: this is exploratory, not an untouched confirmatory test.",
    "Decision calibration is annual OOS conditional on supplied sports OOF predictions; the sports model is never retrained.",
    "No executable prices, fills, settlement ledger, bankroll, staking, limits, slippage or realized strategy ROI are established.",
    "Unit payoffs are retrospective quote/outcome diagnostics under the fixed 12% stake-tax convention, not ledger ROI.",
    "Monthly bootstrap conditions on fitted predictions and observed coverage; it does not refit or repair provenance and excludes no regime risk.",
    "Annual folds share historical training data; confidence intervals are exploratory and not multiplicity-adjusted.",
    "Venue/phase coverage may be selective; open and close are never pooled as independent observations.",
    "Repaired odds are already oriented to GOLGG names; source_sides_swapped is not applied again.",
]


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def code_hashes():
    """Hash the runner and imported local implementation dependencies, not models."""
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT):
                paths.add(path)
    return {str(path): sha256(path) for path in sorted(paths)}


def checked_probabilities(values, name):
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"invalid probabilities: {name}")
    return values


def design(frame, columns):
    return np.column_stack([
        logit(np.clip(checked_probabilities(frame[column], column), EPS, 1 - EPS))
        for column in columns
    ])


def fit_calibrator(x, y):
    """Strictly convex mean BCE + L2/2 * sum(beta**2), no intercept."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or not len(x) or y.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("empty or malformed calibration design")
    if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
        raise ValueError("calibration training requires both binary outcome classes")

    def objective(beta):
        z = x @ beta
        loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * L2 * np.dot(beta, beta)
        gradient = x.T @ (expit(z) - y) / len(y) + L2 * beta
        return float(loss), gradient

    result = minimize(
        objective, np.zeros(x.shape[1]), jac=True, method="BFGS",
        options={"gtol": 1e-8, "maxiter": 2000},
    )
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError(f"decision calibration optimization failed: {result.message}")
    error = float(np.max(np.abs(expit(x @ result.x) + expit(-x @ result.x) - 1)))
    if error > 1e-12:
        raise RuntimeError("decision calibrator violated side symmetry")
    return result.x, {
        "success": bool(result.success), "status": int(result.status),
        "message": str(result.message), "iterations": int(result.nit),
        "function_evaluations": int(result.nfev), "objective": float(result.fun),
        "gradient_inf_norm": float(np.max(np.abs(result.jac))),
        "train_side_symmetry_max_error": error,
    }


def load_aligned(args):
    identities = {key: "string" for key in ("golgg_match_id", "team1_id", "team2_id")}
    pred = pd.read_csv(args.predictions, dtype=identities, usecols=[*METADATA, PREDICTION_COLUMN])
    snap = pd.read_csv(args.snapshots, dtype=identities, usecols=[*METADATA, "team1_name", "team2_name"])
    for name, frame in (("predictions", pred), ("snapshots", snap)):
        if frame[list(METADATA)].isna().any().any() or not frame.golgg_match_id.is_unique:
            raise ValueError(f"{name}: missing metadata or duplicate match IDs")
        dates = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
        if not dates.dt.strftime("%Y-%m-%d").eq(frame.date).all():
            raise ValueError(f"{name}: dates must be canonical ISO days")
        if not frame.y_true.isin([0, 1]).all() or frame.team1_id.eq(frame.team2_id).any():
            raise ValueError(f"{name}: invalid outcomes or identical team IDs")
    missing = ~pred.golgg_match_id.isin(snap.golgg_match_id)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} prediction IDs missing from snapshots")
    frame = pred.merge(snap, on="golgg_match_id", suffixes=("", "__snapshot"), validate="one_to_one")
    for key in METADATA[1:]:
        if not frame[key].eq(frame[f"{key}__snapshot"]).all():
            raise ValueError(f"prediction/snapshot provenance mismatch: {key}")
    frame = frame[[*METADATA, "team1_name", "team2_name", PREDICTION_COLUMN]].copy()
    if frame[["team1_name", "team2_name"]].isna().any().any():
        raise ValueError("snapshot team names missing")
    frame["sports"] = checked_probabilities(frame.pop(PREDICTION_COLUMN), PREDICTION_COLUMN)
    frame = frame.sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    odds = pd.read_csv(args.odds, dtype={"golgg_match_id": "string"}, low_memory=False)
    if odds.golgg_match_id.isna().any():
        raise ValueError("odds contain missing match IDs")
    if "mapping_version" not in odds or not odds.mapping_version.eq("identity-only-v3").all():
        raise ValueError("expected exclusively repaired identity-only-v3 odds")
    return frame, odds, {
        "prediction_rows": len(pred), "snapshot_rows": len(snap), "odds_rows": len(odds),
        "snapshot_rows_without_predictions": int((~snap.golgg_match_id.isin(pred.golgg_match_id)).sum()),
        "prediction_year_counts": {str(k): int(v) for k, v in frame.date.str[:4].value_counts().sort_index().items()},
        "prediction_snapshot_metadata_identical": True,
    }


def phase_frame(frame, odds, phase):
    # Reuse exact identity/date/label and ambiguity rules, with the requested
    # paired STS quote. OPEN eligibility must not depend on averaged CLOSE odds.
    projected = odds.copy()
    for source, target in ((f"odds1_sts_{phase}", "avg_odds_home"), (f"odds2_sts_{phase}", "avg_odds_away")):
        projected[target] = pd.to_numeric(projected[source], errors="raise")
    aligned, audit = attach_market(frame, projected)
    same = aligned.team1_name.eq(aligned.golgg_team1) & aligned.team2_name.eq(aligned.golgg_team2)
    reverse = aligned.team1_name.eq(aligned.golgg_team2) & aligned.team2_name.eq(aligned.golgg_team1)
    ambiguous = aligned.market.notna() & ~(same ^ reverse)
    aligned.loc[ambiguous, ["market", "odds_a", "odds_b"]] = np.nan
    valid = aligned.market.notna()
    audit.update({
        "ambiguous_orientation_rows_excluded": int(ambiguous.sum()),
        "matched_market_rows": int(valid.sum()), "missing_or_unaligned_rows": int((~valid).sum()),
        "reversed_rows": int((reverse & valid).sum()), "phase": phase, "venue": "sts",
        "timing": f"STS {phase}; no certified decision timestamp",
        "pair_eligibility": "phase-specific STS pair, not averaged closing pair",
        "eligible_year_counts": {str(k): int(v) for k, v in aligned.loc[valid, "date"].str[:4].value_counts().sort_index().items()},
    })
    return aligned.loc[valid, [*METADATA, "sports", "market", "odds_a", "odds_b"]].reset_index(drop=True), audit


def oos_predictions(frame, phase):
    results, fits = [], []
    for year in YEARS:
        train = frame.loc[frame.date.lt(f"{year}-01-01")]
        test = frame.loc[frame.date.ge(f"{year}-01-01") & frame.date.lt(f"{year + 1}-01-01")].copy()
        if train.empty or test.empty:
            raise ValueError(f"{phase}/{year}: empty earlier-year train or annual test cohort")
        test["phase"], test["year"] = phase, year
        test["raw_sports"], test["raw_market"] = test.sports, test.market
        for model, columns in FEATURES.items():
            beta, optimization = fit_calibrator(design(train, columns), train.y_true)
            x = design(test, columns)
            p = expit(x @ beta)
            # This check swaps input probabilities, not merely an output logit.
            swapped = test.copy()
            for column in columns:
                swapped[column] = 1 - swapped[column]
            error = float(np.max(np.abs(p + expit(design(swapped, columns) @ beta) - 1)))
            if error > 1e-9:
                raise RuntimeError(f"{phase}/{year}/{model}: side symmetry failed")
            test[model] = p
            fits.append({
                "phase": phase, "test_year": year, "model": model,
                "coefficients": {column: float(value) for column, value in zip(columns, beta)},
                "intercept": 0.0, "l2": L2, "train_n": len(train), "test_n": len(test),
                "train_positive_n": int(train.y_true.sum()), "test_positive_n": int(test.y_true.sum()),
                "train_date_min": str(train.date.min()), "train_date_max": str(train.date.max()),
                "test_date_min": str(test.date.min()), "test_date_max": str(test.date.max()),
                "train_year_counts": {str(k): int(v) for k, v in train.date.str[:4].value_counts().sort_index().items()},
                "sports_input_endpoint_count": int(((train.sports == 0) | (train.sports == 1)).sum()),
                "test_side_symmetry_max_error": error, "optimization": optimization,
            })
        results.append(test)
    return pd.concat(results, ignore_index=True), fits


def paired_uncertainty(values, dates):
    months = pd.to_datetime(dates).dt.to_period("M").nunique()
    if len(values) == 0 or months < 2:
        return {"status": "insufficient observations/months", "n": len(values), "months": int(months), "ci95_low": None, "ci95_high": None}
    return {"status": "ok", **monthly_bootstrap(values, pd.Series(dates).reset_index(drop=True), RESAMPLES)}


def selected_uncertainty(frame, selected, probability_residual, payoff_residual):
    """Resample all eligible months, including months with zero selections."""
    months = pd.to_datetime(frame.date).dt.to_period("M")
    groups = pd.DataFrame({
        "month": months, "count": selected.astype(int),
        "probability": np.where(selected, probability_residual, 0),
        "payoff": np.where(selected, payoff_residual, 0),
    }).groupby("month").sum()
    if not selected.any() or len(groups) < 2:
        return {"status": "no selections" if not selected.any() else "insufficient months", "months": len(groups), "resamples": 0}
    draws = np.random.default_rng(SEED).integers(0, len(groups), size=(RESAMPLES, len(groups)))
    counts = groups["count"].to_numpy()[draws].sum(axis=1)
    valid = counts > 0
    result = {"status": "ok", "months": len(groups), "resamples": RESAMPLES,
              "defined_resamples": int(valid.sum()), "zero_selection_resamples": int((~valid).sum()),
              "conditioning": "percentiles over defined ratios; empty-selection replicates counted, not imputed"}
    for key in ("probability", "payoff"):
        totals = groups[key].to_numpy()[draws].sum(axis=1)
        means = totals[valid] / counts[valid]
        result[key + "_optimism"] = {
            "ci95_low": float(np.quantile(means, .025)) if len(means) else None,
            "ci95_high": float(np.quantile(means, .975)) if len(means) else None,
            "fraction_nonnegative": float(np.mean(means >= 0)) if len(means) else None,
        }
    return result


def summarize(frame, model):
    y, p = frame.y_true.to_numpy(float), frame[model].to_numpy(float)
    if frame.empty:
        return {"eligible_n": 0, "selected_n": 0, "status": "empty cohort"}
    selected, side_a = policy(p, frame.odds_a, frame.odds_b)
    pp, won = np.where(side_a, p, 1 - p), np.where(side_a, y, 1 - y)
    price = np.where(side_a, frame.odds_a, frame.odds_b)
    probability_residual = pp - won
    payoff_residual = (1 - TAX) * price * probability_residual
    result = {
        "eligible_n": len(frame), "selected_n": int(selected.sum()),
        "date_min": str(frame.date.min()), "date_max": str(frame.date.max()),
        "proper_scores_and_calibration": probability_metrics(y, np.clip(p, EPS, 1 - EPS)),
        "selected_reliability": reliability(won[selected], pp[selected], frame.loc[selected, "date"]),
        "selected_reliability_ci_scope": "helper CI resamples selected-containing months only; all-eligible-month CI below is primary",
        "selected_residual_uncertainty": selected_uncertainty(frame, selected, probability_residual, payoff_residual),
    }
    if selected.any():
        result["selected_diagnostics"] = {
            "mean_probability": float(pp[selected].mean()), "win_fraction": float(won[selected].mean()),
            "mean_odds": float(price[selected].mean()),
            "probability_optimism": float(probability_residual[selected].mean()),
            "odds_weighted_return_optimism": float(payoff_residual[selected].mean()),
            "estimated_unit_payoff": float((pp[selected] * price[selected] * (1 - TAX) - 1).mean()),
            "retrospective_realized_unit_payoff": float((won[selected] * price[selected] * (1 - TAX) - 1).mean()),
        }
    else:
        result["selected_diagnostics"] = {"status": "no selections", "n": 0}
    scored_p = np.clip(p, EPS, 1 - EPS)
    market = np.clip(frame.market_calibrated.to_numpy(float), EPS, 1 - EPS)
    result["paired_against_market_calibrated"] = {
        "direction": "model minus calibrated market; negative favors model",
        "log_loss": paired_uncertainty(binary_log_loss_vector(y, scored_p) - binary_log_loss_vector(y, market), frame.date),
        "brier": paired_uncertainty((scored_p - y) ** 2 - (market - y) ** 2, frame.date),
    }
    return result


def risk_slices(frame, model):
    p = frame[model].to_numpy(float)
    _, side_a = policy(p, frame.odds_a, frame.odds_b)
    chosen_odds = np.where(side_a, frame.odds_a, frame.odds_b)
    return {
        "sports_market_disagreement_lt_0.10": np.abs(frame.sports - frame.market) < .10,
        "sports_market_disagreement_ge_0.10": np.abs(frame.sports - frame.market) >= .10,
        "sports_market_opposite_favorites": (frame.sports - .5) * (frame.market - .5) < 0,
        "chosen_odds_lt_3": chosen_odds < 3,
        "chosen_odds_ge_3": chosen_odds >= 3,
    }


def evaluate(predictions):
    summaries, slices, selections = [], [], []
    for phase in PHASES:
        phase_data = predictions.loc[predictions.phase.eq(phase)]
        for period in (*map(str, YEARS), "pooled_2025_2026"):
            frame = phase_data if period.startswith("pooled") else phase_data.loc[phase_data.year.eq(int(period))]
            for model in MODELS:
                header = {"phase": phase, "period": period, "model": model}
                summaries.append({**header, **summarize(frame, model)})
                # Slices are pooled only; cutoffs are fixed in the pre-run protocol.
                if period.startswith("pooled"):
                    for name, mask in risk_slices(frame, model).items():
                        slices.append({**header, "slice": name, **summarize(frame.loc[mask], model)})
        for model in MODELS:
            p = phase_data[model].to_numpy(float)
            selected, side_a = policy(p, phase_data.odds_a, phase_data.odds_b)
            rows = phase_data.loc[selected, ["golgg_match_id", "date", "phase", "year", "y_true"]].copy()
            rows["model"] = model
            rows["side"] = np.where(side_a[selected], "a", "b")
            rows["probability"] = np.where(side_a, p, 1 - p)[selected]
            rows["won"] = np.where(side_a, phase_data.y_true, 1 - phase_data.y_true)[selected]
            rows["odds"] = np.where(side_a, phase_data.odds_a, phase_data.odds_b)[selected]
            rows["estimated_unit_payoff"] = rows.probability * rows.odds * (1 - TAX) - 1
            rows["retrospective_realized_unit_payoff"] = rows.won * rows.odds * (1 - TAX) - 1
            rows["probability_optimism"] = rows.probability - rows.won
            rows["odds_weighted_return_optimism"] = rows.estimated_unit_payoff - rows.retrospective_realized_unit_payoff
            selections.append(rows)
    return summaries, slices, pd.concat(selections, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/artifacts/corrected-historical-reruns-20260908/comparison/predictions.csv")
    parser.add_argument("--snapshots", type=Path, default=ROOT / "data/artifacts/corrected039081-20260908/replay/snapshots.csv")
    parser.add_argument("--odds", type=Path, default=ROOT / "data/artifacts/exp081-full-audit-20260908/identity-only-v3/odds.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory only; existing paths are refused")
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("predictions", "snapshots", "odds", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    inputs = {str(getattr(args, name)): sha256(getattr(args, name)) for name in ("predictions", "snapshots", "odds")}
    code, frozen = code_hashes(), frozen_hashes()
    if (TAX, MIN_EV) != (.12, .05):
        raise RuntimeError("shared policy differs from fixed protocol")
    protocol = {
        "experiment": "exp081-separate-benter-market-calibration", "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does a separate no-intercept sports-plus-market calibration improve OOS proper scores over calibrated STS market?",
        "status": "exploratory_pre_run_protocol", "argv": sys.argv, "input_hashes": inputs,
        "code_hashes": code, "frozen_hashes": frozen, "prediction_column": PREDICTION_COLUMN,
        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "python": sys.version, "platform": platform.platform(), "seed": SEED,
        "phases": {"primary": "sts_close", "sensitivity": "sts_open"},
        "decision_time_certified": False, "eligible_live_rows": 0, "promotion_approved": False,
        "splits": [{"test_year": year, "training": f"all supplied OOF years strictly before {year}"} for year in YEARS],
        "common_cohort": "all five methods use identical train/test valid phase-specific STS pairs; pool years within phase only",
        "eligibility_change_vs_prior_audit": "phase-specific STS eligibility replaces the prior prerequisite of a valid averaged CLOSE pair; overlap/additions are audited per phase",
        "models": list(MODELS), "formula": "sigmoid(beta_s*logit(p_sports)+beta_m*logit(q_market)); omitted term for single-score controls",
        "market_probability": "q_a=(1/odds_a)/(1/odds_a+1/odds_b)=odds_b/(odds_a+odds_b); proportional de-vig, before tax",
        "objective": "mean(logaddexp(0, X@beta)-y*(X@beta)) + (0.001/2)*sum(beta**2)",
        "l2": L2, "intercept": 0, "penalized_coefficients": "all", "coefficient_constraints": "unconstrained",
        "optimizer": {"method": "BFGS", "analytic_gradient": True, "initialization": "zeros", "gtol": 1e-8, "maxiter": 2000, "failure": "raise, no fallback"},
        "logit_probability_clip": EPS, "scoring_probability_clip": EPS,
        "diagnostic_calibration": "shared probability_metrics fits an evaluation-only intercept/slope; these never produce or alter OOS forecasts",
        "selected_reliability_scoring": "existing reliability helper uses its default 1e-15 log-loss clip; selected probabilities and unit payoffs are unmodified",
        "hyperparameter_selection": False, "sports_training": False,
        "policy": {"tax": TAX, "ev_threshold_strictly_greater_than": MIN_EV, "selection": "one maximum-EV side; exact ties abstain"},
        "estimands": {
            "probability_optimism": "mean(selected_probability-selected_outcome)",
            "odds_weighted_return_optimism": "mean((1-tax)*selected_odds*(selected_probability-selected_outcome))",
            "estimated_unit_payoff": "mean((1-tax)*selected_odds*selected_probability-1)",
            "retrospective_realized_unit_payoff": "mean((1-tax)*selected_odds*selected_outcome-1), not ledger ROI",
        },
        "bootstrap": {"unit": "calendar month", "resamples": RESAMPLES, "seed": SEED, "interval": "percentile 95%", "paired_reference": "market_calibrated", "refit": False, "selected_residuals": "resample all eligible months including zero-selection months"},
        "risk_slices": ["abs(raw_sports-raw_market)<0.10", "abs(raw_sports-raw_market)>=0.10", "opposite favorites", "maximum-EV-side odds<3", "maximum-EV-side odds>=3"],
        "limitations": LIMITATIONS,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "protocol.json", protocol)
    try:
        frame, odds, alignment = load_aligned(args)
        prior, prior_audit = attach_market(frame, odds)
        prior_ids = set(prior.loc[prior.market.notna(), "golgg_match_id"])
        alignment["prior_averaged_close_prerequisite"] = prior_audit
        predictions, fits = [], []
        alignment["phases"] = {}
        for phase in PHASES:
            eligible, audit = phase_frame(frame, odds, phase)
            retained = eligible.golgg_match_id.isin(prior_ids)
            audit["prior_audit_eligible_with_same_sts_pair"] = int(retained.sum())
            audit["added_by_removing_averaged_close_prerequisite"] = int((~retained).sum())
            audit["added_year_counts"] = {
                str(k): int(v) for k, v in eligible.loc[~retained, "date"].str[:4].value_counts().sort_index().items()
            }
            alignment["phases"][phase] = audit
            result, phase_fits = oos_predictions(eligible, phase)
            predictions.append(result)
            fits.extend(phase_fits)
        predictions = pd.concat(predictions, ignore_index=True)
        summaries, slices, selections = evaluate(predictions)
        for path, digest in {**inputs, **code}.items():
            if sha256(Path(path)) != digest:
                raise RuntimeError(f"immutable input/code changed during run: {path}")
        if frozen_hashes() != frozen:
            raise RuntimeError("frozen model hashes changed during run")
        predictions.to_csv(args.output_dir / "oos_predictions.csv", index=False)
        selections.to_csv(args.output_dir / "selected_diagnostics.csv", index=False)
        write_json(args.output_dir / "fits.json", fits)
        write_json(args.output_dir / "alignment.json", alignment)
        write_json(args.output_dir / "risk_slices.json", slices)
        summary = {"status": "complete_exploratory", "eligible_live_rows": 0, "promotion_approved": False,
                   "immutable_hashes_verified": True, "oos_rows_by_phase": predictions.groupby("phase").size().to_dict(),
                   "summaries": summaries, "limitations": LIMITATIONS}
        write_json(args.output_dir / "summary.json", summary)
        outputs = {path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}
        write_json(args.output_dir / "manifest.json", {"input_hashes": inputs, "code_hashes": code, "frozen_hashes": frozen, "output_hashes": outputs})
        print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir), "eligible_live_rows": 0, "promotion_approved": False}, allow_nan=False))
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "eligible_live_rows": 0, "promotion_approved": False})
        raise


if __name__ == "__main__":
    main()
