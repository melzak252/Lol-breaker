#!/usr/bin/env python3
"""EXP-083: nested chronological research, no production promotion or betting ROI."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_siamese_architectures import (
    attach_market,
    monthly_bootstrap,
    probability_metrics,
)
from scripts.build_siamese_research_dataset import sha256
from scripts.train_and_tune_siamese_series import (
    build_training_features,
    fit_platt_scaling,
)
from src.analysis.probability_metrics import binary_log_loss_vector
from src.models.correlated_series import (
    fit_correlation,
    fit_map_slope,
    series_probability,
    terminal_score_probability,
)
from src.models.redesign_series import SeriesEstimator, fit_candidates


def temporal_blocks(
    dates: pd.Series,
    year: int,
    protocol: str = "quarter-calibration",
    *,
    train_start: str = "2020-01-01",
) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(dates)
    if protocol not in ("quarter-calibration", "semester-calibration"):
        raise ValueError(f"unknown chronological protocol: {protocol}")
    stopping_end, selection_end = (
        ("04-01", "07-01") if protocol == "semester-calibration" else ("07-01", "10-01")
    )
    boundaries = {
        "train": (train_start, f"{year - 1}-01-01"),
        "stop": (f"{year - 1}-01-01", f"{year - 1}-{stopping_end}"),
        "select": (f"{year - 1}-{stopping_end}", f"{year - 1}-{selection_end}"),
        "calibration": (f"{year - 1}-{selection_end}", f"{year}-01-01"),
        "test": (f"{year}-01-01", f"{year + 1}-01-01"),
    }
    return {
        name: ((dates >= start) & (dates < end)).to_numpy()
        for name, (start, end) in boundaries.items()
    }


def align_legacy_context(legacy: pd.DataFrame, replay: pd.DataFrame):
    """Join on IDs, verify date/format/target, and explicitly project replay sides."""
    if (
        legacy.golgg_match_id.duplicated().any()
        or replay.golgg_match_id.duplicated().any()
    ):
        raise ValueError("duplicate series in legacy/replay join")
    source = replay.set_index("golgg_match_id")
    present = legacy.golgg_match_id.isin(source.index)
    left = legacy.loc[present].reset_index(drop=True).copy()
    right = source.loc[left.golgg_match_id].reset_index()
    same = (left.team1_id == right.team1_id) & (left.team2_id == right.team2_id)
    reverse = (left.team1_id == right.team2_id) & (left.team2_id == right.team1_id)
    date_ok = left.date == right.date
    format_ok = left.best_of == right.best_of
    label_ok = left.y_true.to_numpy() == np.where(
        reverse, 1 - right.y_true, right.y_true
    )
    valid = (same | reverse) & date_ok & format_ok & label_ok
    audit = {
        "legacy_rows": len(legacy),
        "replay_rows": len(replay),
        "missing_replay": int((~present).sum()),
        "identity_mismatch": int((~(same | reverse)).sum()),
        "date_mismatch": int((~date_ok).sum()),
        "format_mismatch": int((~format_ok).sum()),
        "target_mismatch": int((~label_ok).sum()),
        "matched_rows": int(valid.sum()),
        "reversed_rows": int((valid & reverse).sum()),
        "mismatch_counts_may_overlap": True,
    }
    direction = np.where(reverse, -1.0, 1.0)
    for name in right:
        if name.startswith("d_"):
            left["d_replay_" + name[2:]] = right[name].to_numpy() * direction
        elif name.startswith("c_"):
            left["c_replay_" + name[2:]] = right[name].to_numpy()
    left["score_a"] = np.where(reverse, right.score_b, right.score_a)
    left["score_b"] = np.where(reverse, right.score_a, right.score_b)
    left["roster_min_prior_series"] = right.roster_min_prior_series
    for name in ("map_prob_team", "map_prob_player"):
        left[name] = np.where(reverse, 1 - right[name], right[name])
    return left.loc[valid].reset_index(drop=True), audit


def _open(p):
    return np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)


def _controls(frame, masks):
    """Team/player direct series and one-rho generative controls, no current roster."""
    train, stop, cal, test = (
        masks[k] for k in ("train", "stop", "calibration", "test")
    )
    bo = frame.best_of.to_numpy(int)
    y = frame.y_true.to_numpy(float)
    result, metadata = {}, {}
    for side in ("team", "player"):
        z = logit(_open(frame[f"map_prob_{side}"].to_numpy(float)))
        x = np.column_stack((z, z * (bo == 3), z * (bo == 5)))
        scale = x[train].std(axis=0)
        scale[scale == 0] = 1.0
        model = LogisticRegression(C=0.1, fit_intercept=False, max_iter=2000).fit(
            x[train] / scale, y[train]
        )
        slope = fit_platt_scaling(model.decision_function(x[cal] / scale), y[cal])
        result[f"{side}_only"] = _open(
            expit(slope * model.decision_function(x[test] / scale))
        )
        metadata[f"{side}_only"] = {
            "coefficient": model.coef_.ravel().tolist(),
            "scale": scale.tolist(),
            "slope": slope,
            "inputs": [
                f"{side}_map_logit",
                f"{side}_map_logit_bo3",
                f"{side}_map_logit_bo5",
            ],
        }
    z = logit(_open(frame.map_prob_team.to_numpy(float)))
    a, b = frame.score_a.to_numpy(int), frame.score_b.to_numpy(int)
    map_slope = fit_map_slope(z[train], bo[train], a[train], b[train])
    map_p = _open(expit(map_slope * z))
    rho = fit_correlation(map_p[stop], bo[stop], a[stop], b[stop])
    for name, correlation in (("iid_series", 0.0), ("beta_series", rho)):
        raw = _open(series_probability(map_p, bo, correlation))
        slope = fit_platt_scaling(logit(raw[cal]), y[cal])
        result[name] = _open(expit(slope * logit(raw[test])))
        result[name + "__raw"] = raw[test]
        score_p = terminal_score_probability(
            map_p[test], bo[test], a[test], b[test], correlation
        )
        result[name + "__score_log_loss"] = -np.log(score_p)
        metadata[name] = {
            "map_slope": map_slope,
            "rho": correlation,
            "winner_calibration_slope": slope,
            "score_predictions_are_uncalibrated_generative": True,
            "map_slope_fit": "train, IID stopped-score likelihood",
            "rho_fit": "stop, Bo3/5 stopped-score likelihood",
        }
    return result, metadata


def _masks(frame):
    p = frame.fixed_ridge.to_numpy()
    agreement = abs(frame.map_prob_team - frame.map_prob_player)
    masks = {
        "overall": np.ones(len(frame), bool),
        "confidence_heavy": (p >= 0.75) | (p <= 0.25),
        "confidence_moderate": ((p >= 0.6) & (p < 0.75)) | ((p > 0.25) & (p <= 0.4)),
        "confidence_close": (p >= 0.45) & (p <= 0.55),
        "roster_stable": frame.roster_min_prior_series >= 10,
        "roster_rookie": frame.roster_min_prior_series < 10,
        "signals_agree": agreement <= 0.08,
        "signals_disagree": agreement > 0.15,
        "market_common": frame.market.notna(),
        "underdog_3p5_5": frame.odds_a.between(3.5, 5) | frame.odds_b.between(3.5, 5),
        "tier1": frame.competition_tier.isin(["major", "international"]),
    }
    masks.update({f"bo{bo}": frame.best_of == bo for bo in (1, 3, 5)})
    masks.update(
        {
            f"tier_{tier}": frame.competition_tier == tier
            for tier in sorted(frame.competition_tier.unique())
        }
    )
    masks.update(
        {
            f"year_{year}": frame.date.str[:4] == year
            for year in sorted(frame.date.str[:4].unique())
        }
    )
    return masks


def summarize(frame, names, source_blockers):
    masks = _masks(frame)
    metrics = {
        slice_name: {
            name: probability_metrics(frame.loc[mask, "y_true"], frame.loc[mask, name])
            for name in names
        }
        for slice_name, mask in masks.items()
    }
    paired = {}
    for reference in ("fixed_ridge", "ridge"):
        paired[reference] = {}
        base_ll = binary_log_loss_vector(frame.y_true, frame[reference])
        base_bs = (frame[reference] - frame.y_true) ** 2
        for name in names:
            if name != reference:
                paired[reference][name] = {
                    "log_loss": monthly_bootstrap(
                        binary_log_loss_vector(frame.y_true, frame[name]) - base_ll,
                        frame.date,
                    ),
                    "brier": monthly_bootstrap(
                        (frame[name] - frame.y_true) ** 2 - base_bs, frame.date
                    ),
                }
    common = frame[frame.market.notna()]
    market = {
        "n": len(common),
        "benchmark": probability_metrics(common.y_true, common.market),
        "models": {},
    }
    if len(common) > 1:
        for name in names:
            p = common[name]
            market["models"][name] = {
                "delta_log_loss": float(
                    np.mean(
                        binary_log_loss_vector(common.y_true, p)
                        - binary_log_loss_vector(common.y_true, common.market)
                    )
                ),
                "pearson": float(pearsonr(p, common.market).statistic),
                "spearman": float(spearmanr(p, common.market).statistic),
                "mad": float(np.mean(abs(p - common.market))),
                "hybrid_curve_diagnostic_not_selected": {
                    str(alpha): float(
                        binary_log_loss_vector(
                            common.y_true, alpha * p + (1 - alpha) * common.market
                        ).mean()
                    )
                    for alpha in np.linspace(0, 1, 11)
                },
            }
    gates = {}
    baseline = metrics["overall"]["ridge"]
    for name in names:
        if name in ("ridge", "fixed_ridge"):
            continue
        m = metrics["overall"][name]
        tier = metrics["tier1"][name]
        tier_base = metrics["tier1"]["ridge"]
        bad = (
            (frame.odds_a.between(3.5, 5)) & (frame[name] > 0.5) & (frame.market <= 0.5)
        ) | (
            (frame.odds_b.between(3.5, 5)) & (frame[name] < 0.5) & (frame.market >= 0.5)
        )
        gates[name] = {
            "sample_10000": len(frame) >= 10000,
            "ci_superiority_vs_retrained_ridge": paired["ridge"][name]["log_loss"][
                "ci95_high"
            ]
            < 0,
            "tier1_safety": (
                tier["log_loss"] - tier_base["log_loss"] <= 0.002 if tier["n"] else None
            ),
            "ece_not_worse": m["ece10"] <= baseline["ece10"],
            "slope_in_range": (
                0.85 <= m["calibration_slope"] <= 1.15
                if m["calibration_slope"] is not None
                else False
            ),
            "reliability_below_001": m["brier_reliability_binned"] < 0.010,
            "unconfirmed_underdog_favoritism_count": int(bad.sum()),
            "approved": False,
        }
    return {
        "metrics": metrics,
        "paired_deltas": paired,
        "market_diagnostic": market,
        "promotion": {
            "approved": False,
            "gates_against_retrained_control_not_production": gates,
            "blockers": source_blockers,
        },
        "calibration_effect": {
            name: {
                "raw": probability_metrics(frame.y_true, frame[name + "__raw"]),
                "calibrated": metrics["overall"][name],
            }
            for name in names
            if name + "__raw" in frame
        },
        "stopped_score_log_loss": {
            name: float(frame[name + "__score_log_loss"].mean())
            for name in ("iid_series", "beta_series")
            if name + "__score_log_loss" in frame
        },
    }


def run_group(
    label, frame, x, feature_names, n_odd, output, odds, blockers, max_rounds, protocol
):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    y = frame.y_true.to_numpy(float)
    blocks, folds = [], []
    for year in range(2024, pd.Timestamp(frame.date.max()).year + 1):
        masks = temporal_blocks(frame.date, year, protocol)
        if any(not mask.any() for mask in masks.values()):
            raise ValueError(f"{label}/{year}: empty chronological stage")
        folds.append(
            {
                "year": year,
                "stages": {
                    name: {
                        "n": int(mask.sum()),
                        "min": frame.loc[mask, "date"].min(),
                        "max": frame.loc[mask, "date"].max(),
                    }
                    for name, mask in masks.items()
                },
                "models": {},
            }
        )
        args = []
        for name in ("train", "stop", "select", "calibration"):
            args.extend((x[masks[name]], y[masks[name]]))
        fitted = fit_candidates(
            *args,
            n_odd=n_odd,
            feature_names=feature_names,
            seed=83,
            max_rounds=max_rounds,
        )
        part = frame.loc[masks["test"]].copy()
        xt = x[masks["test"]]
        swapped = xt.copy()
        swapped[:, :n_odd] *= -1
        for name, model in fitted.items():
            p = model.predict(xt)
            symmetry = float(np.max(abs(p + model.predict(swapped) - 1)))
            if symmetry > 1e-6:
                raise ValueError(f"{label}/{name}: symmetry failed")
            part[name] = p
            part[name + "__raw"] = model.raw_probability(xt)
            path = output / f"{year}-{name}.joblib"
            model.save(path)
            np.testing.assert_array_equal(
                SeriesEstimator.load(path).predict(xt[:10]), model.predict(xt[:10])
            )
            folds[-1]["models"][name] = {
                **model.metadata,
                "symmetry_max_error": symmetry,
                "artifact_sha256": sha256(path),
            }
            print(
                f"{label} {year} {name}: LL={binary_log_loss_vector(y[masks['test']], p).mean():.6f}",
                flush=True,
            )
        if label == "fresh_replay":
            controls, meta = _controls(frame, masks)
            for name, p in controls.items():
                part[name] = p
            folds[-1]["controls"] = meta
        blocks.append(part)
    result = pd.concat(blocks, ignore_index=True)
    result, market_audit = attach_market(result, odds)
    names = list(fitted)
    if label == "fresh_replay":
        names += ["team_only", "player_only", "iid_series", "beta_series"]
    summary = summarize(result, names, blockers)
    summary.update(
        {
            "group": label,
            "folds": folds,
            "market_audit": market_audit,
            "feature_names": feature_names,
            "n_odd": n_odd,
            "runtime_seconds": time.monotonic() - started,
        }
    )
    result.to_csv(output / "predictions.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return result, summary


def _figures(groups, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label, (frame, _) in groups.items():
        counts = frame.groupby(frame.date.str[:4]).size()
        axes[0].plot(
            counts.index, counts.values, marker="o", label=f"{label} (N={len(frame)})"
        )
    axes[0].set(
        title="EXP-083: evaluated series by year",
        xlabel="Event year",
        ylabel="Completed series",
    )
    axes[0].legend(fontsize=8)
    fresh = groups["fresh_replay"][0]
    for name in ("fixed_ridge", "ridge", "odd_spline", "lightgbm", "stack"):
        bins = np.minimum((fresh[name] * 10).astype(int), 9)
        points = fresh.groupby(bins).agg(
            predicted=(name, "mean"), observed=("y_true", "mean"), n=("y_true", "size")
        )
        points = points[points.n >= 30]
        axes[1].plot(points.predicted, points.observed, marker="o", label=name)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].set(
        title=f"Fresh replay reliability (N={len(fresh)}; bins N≥30)",
        xlabel="Mean predicted P(A)",
        ylabel="Observed A win fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Retrospective sports replay; not production qualification", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--legacy-snapshots", type=Path, required=True)
    parser.add_argument("--legacy-audit", type=Path, required=True)
    parser.add_argument("--odds", type=Path, default=ROOT / "data/odds.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=600)
    parser.add_argument(
        "--protocol",
        choices=("quarter-calibration", "semester-calibration"),
        default="quarter-calibration",
    )
    parser.add_argument(
        "--exp081-artifact",
        type=Path,
        required=True,
        help="explicit archived EXP081 JSON artifact",
    )
    args = parser.parse_args()
    started = time.monotonic()
    identity = {"golgg_match_id": "string", "team1_id": "string", "team2_id": "string"}
    replay_path = args.replay_dir / "snapshots.csv"
    audit = json.loads((args.replay_dir / "audit.json").read_text())
    if sha256(replay_path) != audit["snapshots_sha256"]:
        raise ValueError("replay snapshot hash mismatch")
    legacy_audit = json.loads(args.legacy_audit.read_text())
    replay = (
        pd.read_csv(replay_path, dtype=identity)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    legacy = (
        pd.read_csv(args.legacy_snapshots, dtype=identity)
        .sort_values(["date", "golgg_match_id"])
        .reset_index(drop=True)
    )
    matched, join_audit = align_legacy_context(legacy, replay)
    odds = pd.read_csv(args.odds, dtype={"golgg_match_id": "string"})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    odd = [c for c in replay if c.startswith("d_")]
    even = [c for c in replay if c.startswith("c_")]
    x79, names79 = build_training_features(matched)
    d_context = [c for c in matched if c.startswith("d_replay_")]
    c_context = [c for c in matched if c.startswith("c_replay_")]
    bo = np.column_stack(
        ((matched.best_of == 3).astype(float), (matched.best_of == 5).astype(float))
    )
    experiments = [
        (
            "fresh_replay",
            replay,
            replay[odd + even].to_numpy(float),
            odd + even,
            len(odd),
        ),
        (
            "legacy79",
            matched,
            np.column_stack((x79, bo)),
            ["d_" + name for name in names79] + ["c_bo3", "c_bo5"],
            len(names79),
        ),
        (
            "legacy79_context",
            matched,
            np.column_stack((x79, matched[d_context], bo, matched[c_context])),
            ["d_" + name for name in names79]
            + d_context
            + ["c_bo3", "c_bo5"]
            + c_context,
            len(names79) + len(d_context),
        ),
    ]
    groups = {}
    for label, frame, x, names, n_odd in experiments:
        blockers = list(audit["promotion_blockers"])
        if label != "fresh_replay":
            blockers += legacy_audit["promotion_blockers"]
        groups[label] = run_group(
            label,
            frame,
            x,
            names,
            n_odd,
            args.output_dir / label,
            odds,
            blockers,
            args.max_rounds,
            args.protocol,
        )
    # Existing artifacts are descriptive references only: their historical fit cutoff is not established.
    from src.models.siamese_series import load_artifact
    from src.models.symmetric_series import SymmetricSeriesModel

    legacy_result = groups["legacy79"][0]
    records = legacy_result.to_dict("records")
    frozen = {}
    for name, model in (
        ("stored_exp081", load_artifact(args.exp081_artifact)),
        ("stored_exp078", SymmetricSeriesModel.load_default()),
    ):
        p = np.array(
            [model.predict(row, best_of=int(row["best_of"])) for row in records]
        )
        frozen[name] = probability_metrics(legacy_result.y_true, _open(p))
        legacy_result[name] = p
    legacy_result.to_csv(args.output_dir / "legacy79" / "predictions.csv", index=False)
    # Compare data additions only on identical rows and the same candidate families.
    context = groups["legacy79_context"][0]
    if legacy_result.golgg_match_id.tolist() != context.golgg_match_id.tolist():
        raise ValueError("context ablation changed evaluation cohort")
    context_deltas = {
        name: {
            "log_loss": monthly_bootstrap(
                binary_log_loss_vector(context.y_true, context[name])
                - binary_log_loss_vector(legacy_result.y_true, legacy_result[name]),
                context.date,
            ),
            "brier": monthly_bootstrap(
                (context[name] - context.y_true) ** 2
                - (legacy_result[name] - legacy_result.y_true) ** 2,
                context.date,
            ),
        }
        for name in ("fixed_ridge", "ridge", "odd_spline", "lightgbm", "stack")
    }
    _figures(groups, args.output_dir / "coverage_calibration.svg")
    sources = [
        Path(__file__),
        replay_path,
        args.replay_dir / "audit.json",
        args.legacy_snapshots,
        args.legacy_audit,
        args.odds,
        ROOT / "src/models/redesign_series.py",
        ROOT / "src/models/correlated_series.py",
        ROOT / "src/models/replay_series.py",
        ROOT / "scripts/benchmark_siamese_architectures.py",
        ROOT / "scripts/train_and_tune_siamese_series.py",
        ROOT / "betting_app/models/exp078_symmetric_series_v1.json",
    ]
    artifact_path = args.exp081_artifact.expanduser().resolve()
    artifact_inventory = {
        "path": str(artifact_path),
        "sha256": sha256(artifact_path),
    }
    summary = {
        "experiment": "exp083-model-redesign-v1",
        "scope": "retrospective_nested_chronological_diagnostic",
        "protocol": args.protocol,
        "protocol_selection": (
            "exploratory amendment after quarter-calibration results"
            if args.protocol == "semester-calibration"
            else "locked before candidate evaluation"
        ),
        "plan": "docs/04_experiments/02_produkcja_i_nastepcy/EXP-083_model_redesign.md",
        "runtime_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "package_versions": {
            package: version(package)
            for package in ("scipy", "scikit-learn", "lightgbm", "joblib", "matplotlib")
        },
        "bootstrap_note": "5000 monthly block resamples; p_nonnegative is a bootstrap tail fraction, not automatically a formal hypothesis-test p-value",
        "source_sha256": {str(p): sha256(p) for p in sources},
        "join_audit": join_audit,
        "groups": {label: group[1] for label, group in groups.items()},
        "context_ablation_deltas": context_deltas,
        "stored_artifact_diagnostic": {
            "artifact": artifact_inventory,
            "eligible_as_point_in_time_baseline": False,
            "metrics": frozen,
            "reason": "Artifact fit cutoff/source provenance unverified; numbers cannot establish production superiority",
        },
        "promotion": {"approved": False, "blockers": audit["promotion_blockers"]},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "overall": {k: v[1]["metrics"]["overall"] for k, v in groups.items()},
                "promotion": summary["promotion"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
