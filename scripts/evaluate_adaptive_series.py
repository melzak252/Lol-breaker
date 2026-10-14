#!/usr/bin/env python3
"""EXP-090 evaluation-only diagnostics for frozen odds-free series predictions.

Market prices/probabilities are retrospective report inputs, never fit targets,
model selectors, or actionable prices. Every supplied model keeps the full cohort.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_oddsfree_distillation import (
    CONFIDENCE_EDGES,
    ODDS_EDGES,
    _basic,
    _common_slices,
    _intervals,
    _metrics,
    _paired,
    _reliability,
)

REQUIRED = (
    "golgg_match_id",
    "date",
    "y_true",
    "best_of",
    "tournament",
    "competition_tier",
    "roster_min_prior_series",
    "player_consensus",
    "team_consensus",
    "market",
    "odds_a",
    "odds_b",
    "p__outcome",
    "p__stored_exp081_diagnostic",
)
MIN_CALIBRATION_N = 30
HYBRID_ALPHAS = tuple(index / 10 for index in range(11))


def _validated_frame(predictions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not predictions.columns.is_unique:
        raise ValueError("evaluation column names must be unique")
    missing = sorted(set(REQUIRED) - set(predictions.columns))
    if missing:
        raise ValueError(f"evaluation missing columns: {missing}")
    frame = predictions.reset_index(drop=True).copy()
    if (
        frame.golgg_match_id.isna().any()
        or frame.golgg_match_id.astype(str).duplicated().any()
    ):
        raise ValueError("evaluation requires unique nonmissing series IDs")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame.date.isna().any():
        raise ValueError("evaluation dates must be complete")
    if frame.date.dt.tz is not None:
        frame["date"] = frame.date.dt.tz_localize(None)
    y = frame.y_true.to_numpy(float)
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("evaluation outcomes must be finite binary labels")
    columns = [
        column
        for column in frame
        if isinstance(column, str) and column.startswith("p__")
    ]
    for column in columns:
        p = frame[column].to_numpy(float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(
                f"{column} must supply bounded finite probabilities on the full cohort"
            )
    for column in ("market", "player_consensus", "team_consensus"):
        values = frame[column].to_numpy(float)
        if np.isinf(values).any() or np.any(
            np.isfinite(values) & ((values < 0) | (values > 1))
        ):
            raise ValueError(f"{column} must be missing or a bounded probability")
    for column in ("best_of", "roster_min_prior_series", "odds_a", "odds_b"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame, columns


def _market_diagnostic(y: np.ndarray, p: np.ndarray, market: np.ndarray) -> dict:
    mask = np.isfinite(market)
    y, p, market = y[mask], p[mask], market[mask]
    model_metrics, market_metrics = _basic(y, p), _basic(y, market)
    variable = len(p) >= 2 and np.ptp(p) > 0 and np.ptp(market) > 0
    correlations = {}
    for method in ("pearson", "spearman"):
        value = (
            pd.Series(p).corr(pd.Series(market), method=method) if variable else None
        )
        correlations[method] = (
            float(value) if value is not None and np.isfinite(value) else None
        )
    return {
        "n": len(y),
        "status": "ok" if len(y) else "unavailable_no_market",
        "model": model_metrics,
        "market": market_metrics,
        "log_loss_delta": model_metrics["log_loss"] - market_metrics["log_loss"]
        if len(y)
        else None,
        **correlations,
        "correlation_status": "ok"
        if variable
        else "unavailable_empty_singleton_or_constant",
        "mad": float(np.mean(np.abs(p - market))) if len(y) else None,
        "hybrid_curve": [
            {"alpha": alpha, **_basic(y, alpha * p + (1 - alpha) * market)}
            for alpha in HYBRID_ALPHAS
        ],
        "production_alpha": None,
    }


def _qualification(aggregate: dict, paired: dict, underdogs: dict) -> dict:
    baseline = aggregate["p__outcome"]
    result = {}
    for model, scores in aggregate.items():
        slope, base_slope = scores["calibration_slope"], baseline["calibration_slope"]
        ece, base_ece = scores["ece10"], baseline["ece10"]
        slope_pass = (
            abs(slope - 1) <= abs(base_slope - 1)
            if slope is not None and base_slope is not None
            else None
        )
        ece_pass = ece <= base_ece if ece is not None and base_ece is not None else None
        comparisons = paired.get(model, {})
        ci_gates = {
            name: {
                score: comparison[score]["ci95_high"] < 0
                if comparison[score]["ci95_high"] is not None
                else None
                for score in ("log_loss", "brier")
            }
            for name, comparison in comparisons.items()
        }
        tier = comparisons.get("tier1")
        degradation = {
            score: {
                "delta": tier[score]["mean"] if tier else None,
                "nondegradation_gate": tier[score]["mean"] <= 0
                if tier and tier[score]["mean"] is not None
                else None,
                "promotion_limit": 0.002 if score == "log_loss" else None,
                "promotion_limit_pass": tier[score]["mean"] <= 0.002
                if score == "log_loss" and tier and tier[score]["mean"] is not None
                else None,
            }
            for score in ("log_loss", "brier")
        }
        result[model] = {
            "role": "baseline"
            if model == "p__outcome"
            else "stored_diagnostic"
            if model == "p__stored_exp081_diagnostic"
            else "candidate",
            "sample_gate": {
                "n": scores["n"],
                "minimum_n": 10000,
                "pass": scores["n"] >= 10000,
            },
            "calibration_slope": {
                "value": slope,
                "baseline": base_slope,
                "no_worse_distance_from_one": slope_pass,
                "required_range": [0.85, 1.15],
                "within_required_range": 0.85 <= slope <= 1.15
                if slope is not None
                else None,
            },
            "ece10": {
                "value": ece,
                "baseline": base_ece,
                "no_worse": ece_pass,
                "absolute_limit": 0.03,
                "absolute_limit_pass": ece <= 0.03 if ece is not None else None,
            },
            "brier_reliability": {
                "value": scores["brier_reliability_binned"],
                "limit_exclusive": 0.01,
                "pass": scores["brier_reliability_binned"] < 0.01
                if scores["brier_reliability_binned"] is not None
                else None,
            },
            "tier1_degradation": degradation,
            "bootstrap_ci_gates": ci_gates,
            "bootstrap_status": "reference_not_compared_to_itself"
            if model == "p__outcome"
            else "see_all_and_tier1",
            "underdog_p_gt_05_audit": underdogs[model]["p_gt_05"],
            "underdog_price_350_500_gate": underdogs[model]["price_350_500_gate"],
            "promotion": False,
        }
    return result


def _calibration_plot(rows: list[dict], columns: list[str], path: Path, n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = min(3, len(columns))
    height = (len(columns) + width - 1) // width
    fig, axes = plt.subplots(
        height, width, figsize=(4.3 * width, 3.8 * height), squeeze=False
    )
    for axis, column in zip(axes.flat, columns):
        for orientation, label in (
            ("team_a", "Team A"),
            ("model_favored", "Own favored side"),
        ):
            bins = [
                row
                for row in rows
                if row["model"] == column
                and row["orientation"] == orientation
                and row["n"] >= MIN_CALIBRATION_N
            ]
            axis.plot(
                [row["mean_p"] for row in bins],
                [row["observed_rate"] for row in bins],
                marker="o",
                markersize=3,
                linewidth=1,
                label=label,
            )
        if not any(
            row["model"] == column and row["n"] >= MIN_CALIBRATION_N for row in rows
        ):
            axis.text(
                0.5,
                0.2,
                f"No bins with N >= {MIN_CALIBRATION_N}",
                ha="center",
                fontsize=8,
            )
        axis.plot([0, 1], [0, 1], "k--", linewidth=0.7)
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean predicted series-win probability",
            ylabel="Observed series-win fraction",
        )
        axis.set_title(textwrap.fill(column.removeprefix("p__"), 40), fontsize=8)
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="upper left")
    for axis in list(axes.flat)[len(columns) :]:
        axis.set_visible(False)
    fig.suptitle(
        f"EXP-090 exploratory calibration — N={n}; displayed bins N >= {MIN_CALIBRATION_N}",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "Inspected legacy cohort; no untouched-test or production claim. Orientations overlap. No test calibration applied.",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    try:
        with path.open("xb") as handle:
            fig.savefig(handle, format="svg")
    finally:
        plt.close(fig)


def evaluate(predictions: pd.DataFrame, output_dir: Path) -> dict:
    """Write exclusive, finite EXP-090 diagnostics and return a printable summary.

    All p__ models must cover the same complete unique series cohort. Missing
    diagnostic odds/ratings create explicit slices; they never remove model rows.
    The output directory must not exist, including an existing empty directory.
    """
    frame, columns = _validated_frame(predictions)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to reuse evaluation directory: {output_dir}")
    paths = {
        name: output_dir / filename
        for name, filename in {
            "summary": "summary.json",
            "evaluation": "evaluation.json",
            "metrics": "metrics.csv",
            "paired_comparisons": "paired_comparisons.csv",
            "reliability": "reliability.csv",
            "calibration": "calibration.svg",
        }.items()
    }
    y = frame.y_true.to_numpy(float)
    probabilities = {column: frame[column].to_numpy(float) for column in columns}
    slices = _common_slices(frame)
    slices["tier1"] = frame.competition_tier.isin(("major", "international")).to_numpy()
    for year in (2024, 2025, 2026):
        slices.setdefault(f"year_{year}", (frame.date.dt.year == year).to_numpy())
    odds_a, odds_b = frame.odds_a.to_numpy(float), frame.odds_b.to_numpy(float)
    valid_odds = np.isfinite(odds_a) & np.isfinite(odds_b) & (odds_a > 1) & (odds_b > 1)
    strict_underdog = valid_odds & (odds_a != odds_b)
    market = frame.market.to_numpy(float)
    aggregate, paired, market_diagnostics, underdog_diagnostics = {}, {}, {}, {}
    metric_rows, paired_rows, reliability_rows = [], [], []
    for column, p in probabilities.items():
        for name, mask in slices.items():
            scores = _metrics(y[mask], p[mask])
            metric_rows.append(
                {"model": column, "slice": name, "orientation": "team_a", **scores}
            )
            if name == "all":
                aggregate[column] = scores
        if column != "p__outcome":
            paired[column] = {}
            for name in ("all", "tier1"):
                mask = slices[name]
                comparison = _paired(
                    y[mask],
                    p[mask],
                    probabilities["p__outcome"][mask],
                    frame.date[mask],
                )
                paired[column][name] = comparison
                for score in ("log_loss", "brier"):
                    paired_rows.append(
                        {
                            "model": column,
                            "reference": "p__outcome",
                            "slice": name,
                            "score": score,
                            "n": comparison["n"],
                            "status": comparison["status"],
                            **comparison[score],
                        }
                    )
        favored_a = p >= 0.5
        favored_p, favored_y = (
            np.where(favored_a, p, 1 - p),
            np.where(favored_a, y, 1 - y),
        )
        for orientation, aligned_p, aligned_y in (
            ("team_a", p, y),
            ("model_favored", favored_p, favored_y),
        ):
            reliability_rows.extend(
                {"model": column, "orientation": orientation, **row}
                for row in _reliability(
                    aligned_y, aligned_p, orientation == "model_favored"
                )
            )
        for orientation, side_a in (
            ("market_underdog", odds_a >= odds_b),
            ("market_favorite", odds_a <= odds_b),
            ("model_favored", favored_a),
        ):
            aligned_odds = np.where(side_a, odds_a, odds_b)
            aligned_p, aligned_y = (
                np.where(side_a, p, 1 - p),
                np.where(side_a, y, 1 - y),
            )
            for lower, upper, mask in _intervals(aligned_odds, ODDS_EDGES):
                mask &= valid_odds
                name = (
                    f"odds_{lower:.2f}_{upper:.2f}"
                    if upper is not None
                    else f"odds_ge{lower:.2f}"
                )
                metric_rows.append(
                    {
                        "model": column,
                        "slice": name,
                        "orientation": orientation,
                        "selected_side_a_n": int(np.count_nonzero(mask & side_a)),
                        "selected_side_b_n": int(np.count_nonzero(mask & ~side_a)),
                        **_metrics(aligned_y[mask], aligned_p[mask]),
                    }
                )
        metric_rows.append(
            {
                "model": column,
                "slice": "odds_missing_or_invalid",
                "orientation": "team_a",
                **_metrics(y[~valid_odds], p[~valid_odds]),
            }
        )
        underdog_a = odds_a > odds_b
        underdog_p, underdog_y = (
            np.where(underdog_a, p, 1 - p),
            np.where(underdog_a, y, 1 - y),
        )
        underdog_price = np.where(underdog_a, odds_a, odds_b)
        selected = strict_underdog & (underdog_p >= 0.35) & (underdog_p <= 0.45)
        over_half = strict_underdog & (underdog_p > 0.5)
        price_band = strict_underdog & (underdog_price >= 3.5) & (underdog_price <= 5)
        market_confirmation = np.where(underdog_a, market, 1 - market) > 0.5
        unconfirmed = over_half & price_band & ~market_confirmation
        underdog_diagnostics[column] = {
            "exact_odds_n": int(valid_odds.sum()),
            "strict_underdog_n": int(strict_underdog.sum()),
            "equal_price_n": int(np.count_nonzero(valid_odds & (odds_a == odds_b))),
            "selected_35_45": {
                **_metrics(underdog_y[selected], underdog_p[selected]),
                "minimum_n": MIN_CALIBRATION_N,
                "enough_exact_odds": int(selected.sum()) >= MIN_CALIBRATION_N,
                "evidence_status": "descriptive_sufficient_n"
                if int(selected.sum()) >= MIN_CALIBRATION_N
                else "insufficient_exact_odds_n",
            },
            "p_gt_05": {
                **_metrics(underdog_y[over_half], underdog_p[over_half]),
                "fraction_of_strict_underdogs": float(
                    over_half.sum() / strict_underdog.sum()
                )
                if strict_underdog.any()
                else None,
                "selected_side_a_n": int(np.count_nonzero(over_half & underdog_a)),
                "selected_side_b_n": int(np.count_nonzero(over_half & ~underdog_a)),
            },
            "p_gt_05_price_350_500": _metrics(
                underdog_y[over_half & price_band],
                underdog_p[over_half & price_band],
            ),
            "price_350_500_gate": {
                "observed_price_n": int(price_band.sum()),
                "p_gt_05_n": int((over_half & price_band).sum()),
                "unconfirmed_n": int(unconfirmed.sum()),
                "pass_on_observed_prices": not bool(unconfirmed.any())
                if price_band.any()
                else None,
                "confirmation": "supplied retrospective consensus P(underdog)>.5; missing consensus cannot confirm",
            },
        }
        market_diagnostics[column] = _market_diagnostic(y, p, market)
    qualification = _qualification(aggregate, paired, underdog_diagnostics)
    blockers = [
        "Previously inspected legacy canonical79 cohort; no untouched confirmation cohort.",
        "Legacy source availability, corrected-feature alignment, and roster cutoff provenance remain unverified.",
        "Stored EXP081 is diagnostic only: training provenance is unverified and may overlap this cohort.",
        "Exploratory diagnostics cannot authorize production promotion, live performance, ROI, or bets.",
    ]
    summary = {
        "experiment": "EXP-090",
        "status": "exploratory",
        "n": len(frame),
        "prediction_columns": columns,
        "aggregate": aggregate,
        "paired": paired,
        "market_diagnostics": market_diagnostics,
        "underdog_diagnostics": underdog_diagnostics,
        "qualification": qualification,
        "promotion": False,
        "promotion_blockers": blockers,
        "paths": {name: str(path) for name, path in paths.items()},
    }
    report = {
        **summary,
        "contract": {
            "probability_unit": "P(team A wins the complete series), never a map probability",
            "cohort": "All supplied p__ models score every unique series; missing odds never exclude rows globally",
            "baseline": "p__outcome is the actual odds-free outcome baseline; p__stored_exp081_diagnostic is an unverified/possibly training-overlapping diagnostic, not a qualification reference",
            "selection": "No fitting of predictive parameters, calibration, model selection, or market targets; every supplied frozen model receives the same diagnostics",
            "bootstrap": "Paired candidate-minus-outcome monthly blocks, 5000 resamples, seed 82, two-sided 95% CI; negative favors candidate; no multiplicity adjustment; fewer than two months gives null CI",
            "confidence_edges": list(CONFIDENCE_EDGES),
            "confidence": "Fixed slices use max(p__outcome,1-p__outcome); reliability uses team A and each model's own favored side",
            "tier1": "Combined competition_tier major + international; promotion LL delta limit +.002; strict LL/Brier <=0 also reported diagnostically",
            "calibration_gates": "Required slope [.85,1.15], ECE10<=.03 and no worse than baseline, binned reliability<.010. Slope distance from one is an extra diagnostic, not a replacement for the absolute range. Unavailable fits give null gates",
            "bootstrap_gate": "Global LL upper95%CI<0 is required; Brier and Tier1 CI gates are additional diagnostics, not extra promotion requirements",
            "rating_disagreement": "Absolute difference between player/team MAP-level consensus probabilities, descriptive only; neither is converted into or scored as a series model",
            "odds": "One selected side per series per orientation; flip both p and y for side B; decimal odds must both be finite and >1; price bins [lower,upper), final unbounded; ties choose A for price slices",
            "underdog": "Strict market underdog is the larger exact decimal price; equal-price rows are separately counted and excluded from underdog audits. Selected 35–45% uses that side's model probability inclusively; N>=30 permits descriptive calibration, not confirmation. P>.5 is a retrospective calibration audit, not a betting rule",
            "market": "Common available-market sample only, same oriented team-A p/y; LL delta is model-minus-market. Pearson/Spearman/MAD compare model and supplied market probabilities; constant correlations remain null",
            "hybrid": "Fixed alpha=0,.1,...,1; p=alpha*model+(1-alpha)*market. Diagnostic only; no optimal alpha is selected or promoted",
            "log_loss": "Shared binary_log_loss_vector epsilon=1e-15; Brier uses original probabilities",
            "reliability": "All bins including empty bins are serialized; plots display bins N>=30; orientations overlap and are not independent samples",
        },
        "slices": metric_rows,
        "reliability": reliability_rows,
    }
    encoded_summary = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    encoded_report = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in (("summary", encoded_summary), ("evaluation", encoded_report)):
        with paths[name].open("x", encoding="utf-8") as handle:
            handle.write(content)
    for name, rows in (
        ("metrics", metric_rows),
        ("paired_comparisons", paired_rows),
        ("reliability", reliability_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[name], index=False, mode="x")
    _calibration_plot(reliability_rows, columns, paths["calibration"], len(frame))
    return summary
