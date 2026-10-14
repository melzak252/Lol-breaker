#!/usr/bin/env python3
"""Paired expanding-window architecture diagnostic; never auto-promotes a model.

Hypotheses fixed before this run: linear79, BCE79, focal79 and focal84 with
five antisymmetric disagreement interactions. No market input, hyperparameter
search, financial simulation or retuning on the inspected evaluation cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_siamese_research_dataset import sha256
from scripts.train_and_tune_siamese_series import (
    build_training_features,
    fit_platt_scaling,
    train_single_member,
)
from src.analysis.probability_metrics import binary_log_loss_vector, calculate_ece

VARIANTS = ("linear79", "bce79", "focal79", "focal84")


def augment_disagreement(x: np.ndarray, names: list[str]) -> np.ndarray:
    """Five odd interactions; new inductive bias, NOT new independent evidence."""
    t = x[
        :,
        [names.index(f"team_{s}_logit") for s in ("elo", "gl", "ts", "os", "pl", "tm")],
    ]
    p = x[
        :,
        [
            names.index(f"player_{s}_logit")
            for s in ("elo", "gl", "ts", "os", "pl", "tm")
        ],
    ]
    tm, pm = t.mean(axis=1), p.mean(axis=1)
    d, consensus = tm - pm, (tm + pm) / 2
    extra = np.column_stack(
        (
            consensus * abs(d),
            d * abs(d),
            tm * t.std(axis=1),
            pm * p.std(axis=1),
            consensus * (t * p < 0).mean(axis=1),
        )
    )
    return np.column_stack((x, extra))


def temporal_masks(dates: pd.Series, year: int):
    dates = pd.to_datetime(dates)
    train = (dates < pd.Timestamp(year - 1, 1, 1)).to_numpy()
    calibration = (
        (dates >= pd.Timestamp(year - 1, 1, 1)) & (dates < pd.Timestamp(year, 1, 1))
    ).to_numpy()
    test = (
        (dates >= pd.Timestamp(year, 1, 1)) & (dates < pd.Timestamp(year + 1, 1, 1))
    ).to_numpy()
    return train, calibration, test


def monthly_bootstrap(delta, dates, repetitions=5000):
    delta = np.asarray(delta, dtype=float)
    months = pd.to_datetime(dates).dt.to_period("M")
    if len(delta) != len(months) or not np.isfinite(delta).all():
        raise ValueError("invalid paired deltas")
    groups = (
        pd.DataFrame({"month": months.to_numpy(), "delta": delta})
        .groupby("month")
        .delta.agg(["sum", "count"])
    )
    if len(groups) < 2:
        raise ValueError("at least two months required for block bootstrap")
    if repetitions < 5000:
        raise ValueError("at least 5000 resamples required")
    draws = np.random.default_rng(82).integers(
        0, len(groups), size=(repetitions, len(groups))
    )
    means = groups["sum"].to_numpy()[draws].sum(axis=1) / groups["count"].to_numpy()[
        draws
    ].sum(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "p_nonnegative": float(np.mean(means >= 0)),
        "months": len(groups),
        "resamples": repetitions,
    }


def probability_metrics(y, p):
    """Score probabilities; calibration coefficients are null without a finite,
    identifiable two-parameter logistic MLE (single class, constant predictions,
    or complete/quasi-complete separation). Proper scores remain defined.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if not len(y):
        return {"n": 0}
    if y.shape != p.shape or not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("invalid probability pair")
    bins = np.minimum((p * 10).astype(int), 9)
    reliability = resolution = 0.0
    max_error = 0.0
    for b in np.unique(bins):
        mask = bins == b
        gap = float(p[mask].mean() - y[mask].mean())
        reliability += mask.mean() * gap**2
        resolution += mask.mean() * (y[mask].mean() - y.mean()) ** 2
        max_error = max(max_error, abs(gap))
    slope = intercept = None
    if (
        len(np.unique(y)) == 2
        and p[y == 1].min() < p[y == 0].max()
        and p[y == 0].min() < p[y == 1].max()
    ):
        z = logit(p)

        def objective(theta):
            q = theta[0] + theta[1] * z
            return np.mean(np.logaddexp(0, q) - y * q)

        fit = minimize(objective, [0.0, 1.0], method="BFGS", tol=1e-7)
        if not fit.success:
            raise ValueError(f"diagnostic calibration fit failed: {fit.message}")
        intercept, slope = map(float, fit.x)
    brier = float(np.mean((p - y) ** 2))
    uncertainty = float(y.mean() * (1 - y.mean()))
    return {
        "n": len(y),
        "log_loss": float(binary_log_loss_vector(y, p).mean()),
        "brier": brier,
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "accuracy": float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == y))),
        "ece10": float(calculate_ece(y, p, 10)),
        "mce10": max_error,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "brier_reliability_binned": float(reliability),
        "brier_resolution_binned": float(resolution),
        "brier_uncertainty": uncertainty,
        "brier_binning_residual": float(
            brier - (reliability - resolution + uncertainty)
        ),
    }


def attach_market(frame, odds):
    """Exact historical source identities only; ambiguity excludes every duplicate."""
    odds = odds.copy()
    odds["golgg_match_id"] = odds.golgg_match_id.astype(str)
    dup = odds.golgg_match_id.duplicated(keep=False)
    audit = {"duplicate_odds_rows_excluded": int(dup.sum())}
    selected = odds.loc[
        ~dup,
        [
            "golgg_match_id",
            "golgg_date",
            "golgg_team1",
            "golgg_team2",
            "t1_win",
            "avg_odds_home",
            "avg_odds_away",
        ],
    ]
    common = frame.merge(
        selected, on="golgg_match_id", how="left", validate="one_to_one"
    )
    same = (common.team1_name == common.golgg_team1) & (
        common.team2_name == common.golgg_team2
    )
    reverse = (common.team1_name == common.golgg_team2) & (
        common.team2_name == common.golgg_team1
    )
    aligned = same | reverse
    date_ok = common.date.astype(str).str[:10] == common.golgg_date.astype(str).str[:10]
    label = np.where(
        reverse, 1 - pd.to_numeric(common.t1_win), pd.to_numeric(common.t1_win)
    )
    valid = aligned & date_ok & (label == common.y_true)
    oa = np.where(reverse, common.avg_odds_away, common.avg_odds_home).astype(float)
    ob = np.where(reverse, common.avg_odds_home, common.avg_odds_away).astype(float)
    valid &= np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)
    common["odds_a"] = np.where(valid, oa, np.nan)
    common["odds_b"] = np.where(valid, ob, np.nan)
    common["market"] = np.where(valid, ob / (oa + ob), np.nan)
    audit.update(
        matched_market_rows=int(valid.sum()),
        missing_or_unaligned_rows=int((~valid).sum()),
        reversed_rows=int((reverse & valid).sum()),
        timing="closing benchmark only; no independently recorded quote timestamp",
    )
    return common, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--odds", type=Path, default=ROOT / "data/odds.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--members", type=int, default=5)
    args = parser.parse_args()
    if args.members < 2 or args.epochs < 1:
        parser.error("at least two members and one epoch required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    df = (
        pd.read_csv(args.snapshots, dtype={"golgg_match_id": str})
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    if df.golgg_match_id.duplicated().any():
        raise ValueError("duplicate snapshot series")
    df = df[df.date >= "2020-01-01"].reset_index(drop=True)
    X, names = build_training_features(df)
    X84 = augment_disagreement(X, names)
    np.testing.assert_allclose(augment_disagreement(-X, names), -X84, atol=1e-12)
    y = df.y_true.to_numpy(dtype=float)
    blocks, folds = [], []
    for year in range(2024, pd.Timestamp(df.date.max()).year + 1):
        train, cal, test = temporal_masks(df.date, year)
        if min(train.sum(), cal.sum(), test.sum()) == 0:
            raise ValueError(f"empty chronological partition for {year}")
        part = df.loc[test].copy()
        folds.append(
            {
                "year": year,
                "train_n": int(train.sum()),
                "calibration_n": int(cal.sum()),
                "test_n": int(test.sum()),
                "train_max": df.loc[train, "date"].max(),
                "calibration_min": df.loc[cal, "date"].min(),
                "calibration_max": df.loc[cal, "date"].max(),
                "test_min": df.loc[test, "date"].min(),
                "test_max": df.loc[test, "date"].max(),
            }
        )
        for variant in VARIANTS:
            x = X84 if variant == "focal84" else X
            scales = x[train].std(axis=0)
            scales[scales == 0] = 1  # no centering: preserve f(-x) = -f(x)
            xt, xc, xe = (x[mask] / scales for mask in (train, cal, test))
            if variant == "linear79":
                model = LogisticRegression(C=0.1, fit_intercept=False, max_iter=2000)
                model.fit(xt, y[train])
                slope = fit_platt_scaling(model.decision_function(xc), y[cal])
                z = model.decision_function(xe) * slope
                swapped = model.decision_function(-xe) * slope
                sigma = np.zeros_like(z)
            else:
                members = [
                    train_single_member(
                        xt,
                        y[train],
                        xc,
                        y[cal],
                        epochs=args.epochs,
                        gamma=0.0 if variant == "bce79" else 1.0,
                        seed=42 + 101 * i,
                    )
                    for i in range(args.members)
                ]
                logits = np.stack(
                    [
                        m.forward_anti_symmetric(xe)[0].ravel() * m.platt_slope
                        for m in members
                    ]
                )
                swapped_logits = np.stack(
                    [
                        m.forward_anti_symmetric(-xe)[0].ravel() * m.platt_slope
                        for m in members
                    ]
                )
                z, sigma, swapped = (
                    logits.mean(axis=0),
                    logits.std(axis=0, ddof=0),
                    swapped_logits.mean(axis=0),
                )
            symmetry_error = float(np.max(np.abs(expit(z) + expit(swapped) - 1)))
            if symmetry_error > 1e-6:
                raise ValueError(f"side symmetry failed for {variant}")
            part[variant] = expit(z)
            part[f"{variant}_low_a"] = expit(z - 0.75 * sigma)
            part[f"{variant}_low_b"] = expit(-z - 0.75 * sigma)
            part[f"{variant}_sigma"] = sigma
            folds[-1][f"{variant}_symmetry_max_error"] = symmetry_error
            print(
                f"{year} {variant}: {probability_metrics(y[test], part[variant])['log_loss']:.6f}",
                flush=True,
            )
        blocks.append(part)
    result = pd.concat(blocks, ignore_index=True)
    result, market_audit = attach_market(
        result, pd.read_csv(args.odds, dtype={"golgg_match_id": str})
    )
    # Fix diagnostic cohorts to BCE79 before comparing candidates, never let each
    # candidate select an easier 'coin-flip' or favorite subset for itself.
    pref = result.bce79.to_numpy()
    player = result[
        [f"player_{s}" for s in ("elo", "gl", "ts", "os", "pl", "tm")]
    ].mean(axis=1)
    team = result[[f"team_{s}" for s in ("elo", "gl", "ts", "os", "pl", "tm")]].mean(
        axis=1
    )
    masks = {
        "overall": np.ones(len(result), dtype=bool),
        "confidence_heavy": (pref >= 0.75) | (pref <= 0.25),
        "confidence_moderate": ((pref >= 0.6) & (pref < 0.75))
        | ((pref > 0.25) & (pref <= 0.4)),
        "confidence_close": (pref >= 0.45) & (pref <= 0.55),
        "roster_stable": result.roster_min_prior_series >= 10,
        "roster_rookie": result.roster_min_prior_series < 10,
        "signals_agree": abs(player - team) <= 0.08,
        "signals_disagree": abs(player - team) > 0.15,
        "market_common": result.market.notna(),
        "odds_underdog_3p5_5": result.odds_a.between(3.5, 5)
        | result.odds_b.between(3.5, 5),
    }
    masks.update({f"bo{bo}": result.best_of == bo for bo in (1, 3, 5)})
    masks.update(
        {
            f"tier_{tier}": result.competition_tier == tier
            for tier in result.competition_tier.unique()
        }
    )
    metrics = {
        name: {
            v: probability_metrics(result.loc[mask, "y_true"], result.loc[mask, v])
            for v in VARIANTS
        }
        for name, mask in masks.items()
    }
    deltas = {}
    for reference in ("linear79", "bce79"):
        deltas[reference] = {}
        base_ll = binary_log_loss_vector(result.y_true, result[reference])
        base_bs = (result[reference] - result.y_true) ** 2
        for v in VARIANTS:
            if v != reference:
                deltas[reference][v] = {
                    "log_loss": monthly_bootstrap(
                        binary_log_loss_vector(result.y_true, result[v]) - base_ll,
                        result.date,
                    ),
                    "brier": monthly_bootstrap(
                        (result[v] - result.y_true) ** 2 - base_bs, result.date
                    ),
                }
    market = result[result.market.notna()]
    market_stats = {
        "n": len(market),
        "benchmark": probability_metrics(market.y_true, market.market),
        "models": {},
    }
    if len(market):
        for v in VARIANTS:
            p = market[v]
            market_stats["models"][v] = {
                "delta_log_loss": float(
                    np.mean(
                        binary_log_loss_vector(market.y_true, p)
                        - binary_log_loss_vector(market.y_true, market.market)
                    )
                ),
                "pearson": float(pearsonr(p, market.market).statistic),
                "spearman": float(spearmanr(p, market.market).statistic),
                "mad": float(np.mean(abs(p - market.market))),
                "hybrid_curve_diagnostic_not_tuned": {
                    str(a): float(
                        binary_log_loss_vector(
                            market.y_true, a * p + (1 - a) * market.market
                        ).mean()
                    )
                    for a in np.linspace(0, 1, 11)
                },
            }
    source_audit = json.loads(args.audit.read_text())
    summary = {
        "experiment": "siamese-integrity-v1",
        "scope": "retrospective_architecture_diagnostic_not_promotion",
        "protocol": {
            "folds": folds,
            "epochs": args.epochs,
            "members": args.members,
            "seeds": [42 + 101 * i for i in range(args.members)],
            "calibration": "preceding calendar year, disjoint from training and evaluation",
            "tuning": "none; no test-based selection",
            "feature_names79": names,
            "input_policy": "all complete series, not selected by market availability",
            "p_low": "heuristic sigmoid(mean_logit +/- side - 0.75*population_sd), not confidence guarantee",
        },
        "dataset_audit": source_audit,
        "metrics": metrics,
        "paired_deltas": deltas,
        "market_audit": market_audit,
        "market_diagnostic": market_stats,
        "promotion": {
            "approved": False,
            "sample_gate": len(result) >= 10000,
            "reasons": source_audit["promotion_blockers"]
            + [
                "no point-in-time locked EXP039/operational baseline predictions on this cohort",
                "no event-time ledger; no executable financial claims",
            ],
        },
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sha256": {
                str(p): sha256(p)
                for p in (
                    args.snapshots,
                    args.audit,
                    args.odds,
                    Path(__file__),
                    ROOT / "scripts/train_and_tune_siamese_series.py",
                    ROOT / "src/models/symmetric_series.py",
                )
            },
        },
    }
    result.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {"overall": metrics["overall"], "promotion": summary["promotion"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
