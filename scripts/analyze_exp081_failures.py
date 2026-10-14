#!/usr/bin/env python3
"""Retrospective EXP081 failure audit; not a model promotion or betting backtest.

Predeclared diagnostics: confidence, format, tier/family/year, team-player
conflict, W20 coverage, prior-roster experience/freshness, ensemble spread,
input extrapolation, and market disagreement. Subgroups are exploratory;
monthly intervals do not correct multiple/adaptive comparisons.

Controlled input ablations use fixed ridge C=.1, train through Y-2 and
calibration July-December Y-1, then score Y. No outcome selects a feature set.
These test available information under a linear learner, not a causal account
of the frozen MLP. Legacy feature and artifact provenance remain unverified.
No database access, scraping, production writes, or artifact replacement.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import sys
import time
import warnings

import numpy as np
import pandas as pd
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
from src.models.competition_tiers import classify_competition
from src.models.siamese_series import SiameseSeriesModel, load_artifact

SYSTEMS = ("elo", "gl", "ts", "os", "pl", "tm")
EPS = np.finfo(float).eps


def bootstrap(delta, dates):
    if pd.to_datetime(dates).dt.to_period("M").nunique() < 2:
        return None
    return monthly_bootstrap(delta, dates)


def summarize(frame):
    if frame.empty:
        return {"n": 0}
    p, y = frame.stored_exp081.to_numpy(), frame.y_true.to_numpy()
    confidence = np.maximum(p, 1 - p)
    correct = np.where(p == 0.5, 0.5, (p > 0.5) == y)
    result = {
        **probability_metrics(y, p),
        "expected_error_fraction": float(np.mean(1 - confidence)),
        "observed_error_fraction": float(np.mean(1 - correct)),
        "overconfidence_gap": float(np.mean(confidence - correct)),
        "ll_excess_vs_ridge": bootstrap(
            binary_log_loss_vector(y, p) - binary_log_loss_vector(y, frame.fixed_ridge),
            frame.date,
        ),
        "market_n": int(frame.market.notna().sum()),
    }
    common = frame[frame.market.notna()]
    if len(common):
        result["market_log_loss"] = float(
            binary_log_loss_vector(common.y_true, common.market).mean()
        )
        result["model_market_common_log_loss"] = float(
            binary_log_loss_vector(common.y_true, common.stored_exp081).mean()
        )
        result["ll_excess_vs_market"] = bootstrap(
            binary_log_loss_vector(common.y_true, common.stored_exp081)
            - binary_log_loss_vector(common.y_true, common.market),
            common.date,
        )
    return result


def annotate(frame, model):
    frame = frame.copy()
    x, names = build_training_features(frame)
    if tuple(names) != model.feature_names:
        raise ValueError("canonical feature order differs from artifact")
    scaled = (x - model.means) / model.scales
    logits = np.array(
        [
            [member.forward_anti_symmetric(row) for member in model.members]
            for row in scaled
        ]
    )
    np.testing.assert_allclose(
        expit(logits.mean(axis=1)), frame.stored_exp081, atol=1e-12, rtol=0
    )
    frame["sigma_z"] = logits.std(axis=1)
    frame["max_abs_scaled_input"] = np.max(abs(scaled), axis=1)
    frame["confidence"] = np.maximum(frame.stored_exp081, 1 - frame.stored_exp081)
    for side in ("team", "player"):
        raw = frame[[f"{side}_{s}" for s in SYSTEMS]].to_numpy()
        if not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
            raise ValueError("invalid rating probabilities")
        frame[f"{side}_consensus"] = expit(
            logit(np.clip(raw, 0.001, 0.999)).mean(axis=1)
        )
    frame["rating_gap"] = abs(frame.team_consensus - frame.player_consensus)
    frame["rating_conflict"] = (frame.team_consensus - 0.5) * (
        frame.player_consensus - 0.5
    ) < 0
    frame["fresh_rating_gap"] = abs(frame.map_prob_team - frame.map_prob_player)
    frame["w20_min_games"] = frame[["history_games_1", "history_games_2"]].min(axis=1)
    frame["rest_max_days"] = frame[["days_since_last_1", "days_since_last_2"]].max(
        axis=1
    )
    # Replay context is (A+B)/2, odd part A-B. Recover maximum without current roster.
    age = frame.c_replay_roster_age_log1p + abs(frame.d_replay_roster_age_log1p) / 2
    frame["roster_proxy_max_age_days"] = np.expm1(age)
    known = frame.c_replay_roster_known - abs(frame.d_replay_roster_known) / 2
    frame["both_prior_rosters_known"] = known > 0.5
    frame["family"] = [
        classify_competition(t, pd.Timestamp(d).date()).family
        for t, d in zip(frame.tournament, frame.date)
    ]
    frame["year"] = frame.date.str[:4]
    frame["ll081"] = binary_log_loss_vector(frame.y_true, frame.stored_exp081)
    frame["ll_ridge"] = binary_log_loss_vector(frame.y_true, frame.fixed_ridge)
    frame["ll_market"] = np.nan
    common = frame.market.notna()
    frame.loc[common, "ll_market"] = binary_log_loss_vector(
        frame.loc[common, "y_true"], frame.loc[common, "market"]
    )
    return frame


def slices(frame):
    p = frame.stored_exp081
    sigma_cuts = frame.sigma_z.quantile([0.25, 0.5, 0.75]).to_list()
    masks = {
        "overall": np.ones(len(frame), bool),
        "confidence_50_60": frame.confidence < 0.6,
        "confidence_60_75": frame.confidence.between(0.6, 0.75, inclusive="left"),
        "confidence_75_90": frame.confidence.between(0.75, 0.9, inclusive="left"),
        "confidence_90_plus": frame.confidence >= 0.9,
        "legacy_gap_le008": frame.rating_gap <= 0.08,
        "legacy_gap_008_015": frame.rating_gap.between(0.08, 0.15, inclusive="right"),
        "legacy_gap_015_030": frame.rating_gap.between(0.15, 0.3, inclusive="right"),
        "legacy_gap_gt030": frame.rating_gap > 0.3,
        "legacy_opposite_favorites": frame.rating_conflict,
        "legacy_opposite_gap_gt015": frame.rating_conflict & (frame.rating_gap > 0.15),
        "fresh_gap_le008": frame.fresh_rating_gap <= 0.08,
        "fresh_gap_gt015": frame.fresh_rating_gap > 0.15,
        "w20_lt10": frame.w20_min_games < 10,
        "w20_10_19": frame.w20_min_games.between(10, 20, inclusive="left"),
        "w20_complete20": frame.w20_min_games >= 20,
        "prior_roster_rookie": frame.roster_min_prior_series < 10,
        "prior_roster_experienced": frame.roster_min_prior_series >= 10,
        "prior_roster_unknown": ~frame.both_prior_rosters_known,
        "prior_roster_age_le30": frame.both_prior_rosters_known
        & (frame.roster_proxy_max_age_days <= 30),
        "prior_roster_age_gt30": frame.both_prior_rosters_known
        & (frame.roster_proxy_max_age_days > 30),
        "rest_gt30": frame.rest_max_days > 30,
        "rest_le30": frame.rest_max_days <= 30,
        "input_extrapolation_gt8sd": frame.max_abs_scaled_input > 8,
        "input_extrapolation_le4sd": frame.max_abs_scaled_input <= 4,
        "market_agreement_le008": abs(p - frame.market) <= 0.08,
        "market_disagreement_gt015": abs(p - frame.market) > 0.15,
        "market_underdog_3p5_5_model_favorite": (
            (frame.odds_a.between(3.5, 5)) & (p > 0.5) & (frame.market < 0.5)
        )
        | ((frame.odds_b.between(3.5, 5)) & (p < 0.5) & (frame.market > 0.5)),
    }
    for i in range(4):
        lo = -np.inf if i == 0 else sigma_cuts[i - 1]
        hi = np.inf if i == 3 else sigma_cuts[i]
        masks[f"member_spread_q{i + 1}"] = frame.sigma_z.between(
            lo, hi, inclusive="right"
        )
    for key in ("year", "best_of", "competition_tier", "family"):
        for value in sorted(frame[key].unique()):
            masks[f"{key}:{value}"] = frame[key] == value
    result = {name: summarize(frame.loc[mask]) for name, mask in masks.items()}
    # Stability by year prevents a league/format mixture from becoming a causal claim.
    stable = {}
    for name in (
        "legacy_opposite_gap_gt015",
        "market_disagreement_gt015",
        "prior_roster_age_gt30",
        "w20_lt10",
        "confidence_90_plus",
    ):
        stable[name] = {
            year: summarize(frame.loc[masks[name] & (frame.year == year)])
            for year in sorted(frame.year.unique())
        }
    conflict = frame[frame.rating_conflict & (frame.rating_gap > 0.15)]
    choices = {}
    for label, signal in [
        ("team", "team_consensus"),
        ("player", "player_consensus"),
        ("model", "stored_exp081"),
    ]:
        choices[label] = {
            "n": len(conflict),
            "accuracy": float(np.mean((conflict[signal] > 0.5) == conflict.y_true))
            if len(conflict)
            else None,
        }
    for side in ("team", "player"):
        mask = (conflict.stored_exp081 > 0.5) == (conflict[f"{side}_consensus"] > 0.5)
        choices["model_follows_" + side] = summarize(conflict.loc[mask])
    return result, stable, choices, sigma_cuts


def input_variants(full):
    x, names = build_training_features(full)
    names = list(names)
    w20 = np.array([name.startswith("w20_") for name in names])
    team = np.array([name.startswith("team_") for name in names])
    player = np.array([name.startswith("player_") for name in names])
    rest = np.array([name == "rest_days_diff" for name in names])
    variants = {"canonical79": (x, names)}
    for label, mask in [
        ("ratings_without_w20", ~w20),
        ("w20_and_rest", w20 | rest),
        ("team_w20_rest", team | w20 | rest),
        ("player_w20_rest", player | w20 | rest),
    ]:
        variants[label] = (
            x[:, mask],
            [name for name, keep in zip(names, mask) if keep],
        )
    t = logit(
        np.clip(full[[f"team_{s}" for s in SYSTEMS]].to_numpy(), 0.001, 0.999)
    ).mean(axis=1)
    p = logit(
        np.clip(full[[f"player_{s}" for s in SYSTEMS]].to_numpy(), 0.001, 0.999)
    ).mean(axis=1)
    # Even reliability modulates odd evidence; it must not be a free side intercept.
    context_names = [
        "c_replay_roster_known",
        "c_replay_roster_age_log1p",
        "c_replay_roster_ambiguous",
        "c_replay_team_rest_log1p",
        "c_replay_player_series_log1p",
        "c_replay_team_rd",
        "c_replay_player_rd",
        "c_replay_history_games",
    ]
    context = full[context_names].to_numpy()
    additions = np.column_stack([context * t[:, None], context * p[:, None]])
    addition_names = [
        f"{side}_logit_x_{name}"
        for side in ("team", "player")
        for name in context_names
    ]
    variants["plus_reliability16"] = (
        np.column_stack((x, additions)),
        names + addition_names,
    )
    recent_names = ["d_replay_map_win_short_long", "d_replay_residual_short_long"]
    recent = full[recent_names].to_numpy()
    variants["plus_recent_form2"] = (np.column_stack((x, recent)), names + recent_names)
    gap = abs(expit(t) - expit(p))
    variants["plus_disagreement2"] = (
        np.column_stack((x, t * gap, p * gap)),
        names + ["team_logit_x_gap", "player_logit_x_gap"],
    )
    return variants


def ablations(full, evaluation, output):
    variants = input_variants(full)
    y = full.y_true.to_numpy()
    rows, metadata = [], []
    for year in (2024, 2025, 2026):
        masks = temporal_blocks(full.date, year, protocol="semester-calibration")
        train, cal, test = (masks[k] for k in ("train", "calibration", "test"))
        if not all(m.any() for m in (train, cal, test)):
            raise ValueError("empty ablation stage")
        part = full.loc[test, ["golgg_match_id", "date", "y_true"]].copy()
        for name, (x, feature_names) in variants.items():
            if not np.isfinite(x).all():
                raise ValueError(f"nonfinite features: {name}")
            scaler = StandardScaler(with_mean=False).fit(x[train])
            model = LogisticRegression(
                C=0.1,
                fit_intercept=False,
                solver="lbfgs",
                max_iter=10000,
                tol=1e-8,
                random_state=83,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(scaler.transform(x[train]), y[train])
            # Match EXP083 calibration of the actual bounded raw probability,
            # including its expit/logit round trip (optimizer-sensitive at 1e-8).
            cal_z = logit(
                np.clip(
                    expit(model.decision_function(scaler.transform(x[cal]))),
                    EPS,
                    1 - EPS,
                )
            )
            slope = fit_platt_scaling(cal_z, y[cal])
            z = logit(
                np.clip(
                    expit(model.decision_function(scaler.transform(x[test]))),
                    EPS,
                    1 - EPS,
                )
            )
            probabilities = np.clip(expit(slope * z), EPS, 1 - EPS)
            reverse_z = logit(
                np.clip(
                    expit(model.decision_function(scaler.transform(-x[test]))),
                    EPS,
                    1 - EPS,
                )
            )
            reverse = expit(slope * reverse_z)
            np.testing.assert_allclose(probabilities + reverse, 1, atol=1e-12, rtol=0)
            part[name] = probabilities
            metadata.append(
                {
                    "year": year,
                    "variant": name,
                    "feature_names": feature_names,
                    "coefficient": model.coef_[0].tolist(),
                    "scale": scaler.scale_.tolist(),
                    "slope": slope,
                    "stages": {
                        k: {
                            "n": int(masks[k].sum()),
                            "min": full.loc[masks[k], "date"].min(),
                            "max": full.loc[masks[k], "date"].max(),
                        }
                        for k in ("train", "calibration", "test")
                    },
                }
            )
        rows.append(part)
    predictions = pd.concat(rows, ignore_index=True)
    if predictions.golgg_match_id.tolist() != evaluation.golgg_match_id.tolist():
        raise ValueError("ablation changes evaluation cohort/order")
    np.testing.assert_allclose(
        predictions.canonical79, evaluation.fixed_ridge, atol=1e-10, rtol=0
    )
    predictions.to_csv(output / "input_ablation_predictions.csv", index=False)
    results = {}
    for name in variants:
        results[name] = {
            "features": len(variants[name][1]),
            "metrics": probability_metrics(predictions.y_true, predictions[name]),
            "paired_vs_canonical79": bootstrap(
                binary_log_loss_vector(predictions.y_true, predictions[name])
                - binary_log_loss_vector(predictions.y_true, predictions.canonical79),
                predictions.date,
            ),
            "year": {
                str(year): probability_metrics(
                    predictions.loc[
                        predictions.date.str.startswith(str(year)), "y_true"
                    ],
                    predictions.loc[predictions.date.str.startswith(str(year)), name],
                )
                for year in (2024, 2025, 2026)
            },
        }
    return results, metadata, predictions


def followup_diagnostics(frame, predictions):
    """Exploratory drill-down selected AFTER first-pass family results."""
    results = {}
    for family in ("LPL", "LCK CL", "LFL Div2", "CBLOL", "LCS", "Worlds"):
        part = frame[frame.family == family]
        correct = np.where(
            part.stored_exp081 == 0.5, 0.5, (part.stored_exp081 > 0.5) == part.y_true
        )
        controls = {}
        for name in ("team_w20_rest", "player_w20_rest", "plus_reliability16"):
            controls[name] = bootstrap(
                binary_log_loss_vector(part.y_true, predictions.loc[part.index, name])
                - binary_log_loss_vector(
                    part.y_true, predictions.loc[part.index, "canonical79"]
                ),
                part.date,
            )
        results[family] = {
            "favorite_overconfidence_ci": bootstrap(
                part.confidence - correct, part.date
            ),
            "year": {
                str(year): summarize(group) for year, group in part.groupby("year")
            },
            "format": {
                str(bo): summarize(group) for bo, group in part.groupby("best_of")
            },
            "input_ablation_delta_LL": controls,
        }
    for name in (
        "canonical79",
        "ratings_without_w20",
        "w20_and_rest",
        "team_w20_rest",
        "player_w20_rest",
        "plus_reliability16",
        "plus_recent_form2",
        "plus_disagreement2",
    ):
        results["paired_brier:" + name] = bootstrap(
            (predictions[name] - predictions.y_true) ** 2
            - (predictions.canonical79 - predictions.y_true) ** 2,
            predictions.date,
        )
    return {
        "selected_after_first_pass": True,
        "multiple_comparisons_corrected": False,
        "results": results,
    }


def plots(frame, stats, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    group_names = [k for k in stats if k.startswith(("best_of:", "competition_tier:"))]
    values = [stats[k]["ll_excess_vs_market"] for k in group_names]
    keep = [
        (k, v)
        for k, v in zip(group_names, values)
        if v is not None and stats[k]["market_n"] >= 50
    ]
    x = [v["mean"] for _, v in keep]
    axes[0].errorbar(
        x,
        range(len(keep)),
        xerr=[
            [v["mean"] - v["ci95_low"] for _, v in keep],
            [v["ci95_high"] - v["mean"] for _, v in keep],
        ],
        fmt="o",
    )
    axes[0].set_yticks(
        range(len(keep)), [f"{k} (N={stats[k]['market_n']})" for k, _ in keep]
    )
    axes[0].axvline(0, color="gray", linestyle="--")
    axes[0].set(
        title="EXP081 minus closing-market LogLoss (N≥50)",
        xlabel="Paired mean and monthly 95% CI",
    )
    bins = pd.cut(frame.confidence, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    groups = (
        frame.assign(correct=(frame.stored_exp081 > 0.5) == frame.y_true)
        .groupby(bins, observed=True)
        .agg(
            confidence=("confidence", "mean"),
            observed=("correct", "mean"),
            n=("correct", "size"),
        )
    )
    axes[1].plot(groups.confidence, groups.observed, "o-")
    axes[1].plot([0.5, 1], [0.5, 1], "k--")
    for _, r in groups.iterrows():
        axes[1].annotate(
            f"N={int(r.n)}",
            (r.confidence, r.observed),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set(
        title=f"Predicted favorite reliability (N={len(frame)})",
        xlabel="Mean predicted win probability",
        ylabel="Observed win fraction",
        xlim=(0.5, 1),
        ylim=(0.5, 1),
    )
    labels = [
        "legacy_gap_le008",
        "legacy_gap_008_015",
        "legacy_gap_015_030",
        "legacy_gap_gt030",
    ]
    axes[2].bar(range(4), [stats[k]["log_loss"] for k in labels])
    axes[2].set_xticks(
        range(4),
        [
            f"{label}\nN={stats[k]['n']}"
            for label, k in zip(["≤.08", ".08–.15", ".15–.30", ">.30"], labels)
        ],
    )
    axes[2].set(
        title="Team–player rating disagreement",
        xlabel="Absolute difference of map-probability consensus",
        ylabel="EXP081 series LogLoss (not difficulty-adjusted)",
    )
    fig.suptitle(
        "Retrospective artifact diagnosis; no proven training cutoff or executable betting claim"
    )
    fig.tight_layout()
    fig.savefig(output / "failure_diagnostics.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=ROOT / "data/artifacts/model-redesign-v1/benchmark-semester",
    )
    parser.add_argument(
        "--legacy-snapshots",
        type=Path,
        default=ROOT / "data/artifacts/siamese-integrity-v1/snapshots.csv",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=ROOT / "data/artifacts/model-redesign-v1/replay",
    )
    parser.add_argument(
        "--exp081-artifact",
        type=Path,
        required=True,
        help="explicit archived EXP081 JSON artifact",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    identity = {"golgg_match_id": "string", "team1_id": "string", "team2_id": "string"}
    prediction_path = args.benchmark_dir / "legacy79/predictions.csv"
    frame = (
        pd.read_csv(prediction_path, dtype=identity, low_memory=False)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    if frame.golgg_match_id.duplicated().any() or frame.stored_exp081.isna().any():
        raise ValueError("missing/duplicate frozen predictions")
    model = load_artifact(args.exp081_artifact)
    frame = annotate(frame, model)
    stats, stability, choices, sigma_cuts = slices(frame)
    legacy = (
        pd.read_csv(args.legacy_snapshots, dtype=identity, low_memory=False)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    replay_path = args.replay_dir / "snapshots.csv"
    replay_audit = json.loads((args.replay_dir / "audit.json").read_text())
    if sha256(replay_path) != replay_audit["snapshots_sha256"]:
        raise ValueError("replay hash mismatch")
    replay = pd.read_csv(replay_path, dtype=identity, low_memory=False)
    full, join_audit = align_legacy_context(legacy, replay)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs, fit_metadata, predictions = ablations(full, frame, args.output_dir)
    # Team tables are descriptive; games involve two teams, so rows are not independent.
    team_rows = pd.concat(
        [
            frame.assign(team_id=frame[col], team_name=frame[name])
            for col, name in [("team1_id", "team1_name"), ("team2_id", "team2_name")]
        ],
        ignore_index=True,
    )
    teams = {
        str(tid): {"name": part.team_name.iloc[-1], **summarize(part)}
        for tid, part in team_rows.groupby("team_id")
        if len(part) >= 80
    }
    # Preserve failures and successes: examples are not the analysis cohort.
    examples = frame[
        (frame.confidence >= 0.75) & ((frame.stored_exp081 > 0.5) != frame.y_true)
    ].sort_values("ll081", ascending=False)
    cols = [
        "golgg_match_id",
        "date",
        "team1_name",
        "team2_name",
        "tournament",
        "best_of",
        "y_true",
        "stored_exp081",
        "fixed_ridge",
        "market",
        "team_consensus",
        "player_consensus",
        "rating_gap",
        "sigma_z",
        "w20_min_games",
        "roster_min_prior_series",
        "roster_proxy_max_age_days",
        "ll081",
        "ll_market",
    ]
    examples[cols].head(30).to_csv(
        args.output_dir / "worst_confident_errors.csv", index=False
    )
    frame.to_csv(args.output_dir / "annotated_predictions.csv", index=False)
    plots(frame, stats, args.output_dir)
    source_paths = [
        args.exp081_artifact,
        prediction_path,
        args.legacy_snapshots,
        replay_path,
        ROOT / "src/models/symmetric_series.py",
        ROOT / "src/models/siamese_series.py",
        ROOT / "scripts/benchmark_model_redesign.py",
        ROOT / "scripts/benchmark_siamese_architectures.py",
        ROOT / "scripts/train_and_tune_siamese_series.py",
    ]
    report = {
        "experiment": "exp081-failure-audit-v1",
        "scope": "retrospective exploratory diagnosis; no promotion",
        "n": len(frame),
        "date_min": frame.date.min(),
        "date_max": frame.date.max(),
        "slices": stats,
        "stability_by_year": stability,
        "conflicting_rating_choices": choices,
        "ensemble_spread_quartiles": sigma_cuts,
        "team_diagnostics_min80": teams,
        "input_ablations": inputs,
        "ablation_fit_metadata": fit_metadata,
        "join_audit": join_audit,
        "runtime_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "python": platform.python_version(),
        "versions": {
            p: version(p)
            for p in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")
        },
        "source_sha256": {str(p): sha256(p) for p in source_paths},
        "limitations": [
            "Frozen EXP081 training cutoff and legacy rating source provenance are unverified.",
            "Closing market is diagnostic only; no quote timestamp audit.",
            "Slices and repeated historical cohort are exploratory, no multiple-comparison correction.",
            "High loss may reflect harder games, not excess model error; use paired controls.",
            "Rating ablations use a fixed linear estimator; not proof of causal MLP failure.",
            "Lagged roster is a prior-day proxy, not the actual announced target lineup.",
            "No new architecture, production promotion, database writes or betting return claim.",
        ],
    }
    report["exploratory_followup"] = followup_diagnostics(frame, predictions)
    report["runtime_seconds"] = time.monotonic() - started
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "overall": stats["overall"],
                "conflicting_ratings": choices,
                "input_ablation_LL": {
                    k: v["metrics"]["log_loss"] for k, v in inputs.items()
                },
                "runtime_seconds": report["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
