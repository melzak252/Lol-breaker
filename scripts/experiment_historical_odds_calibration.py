#!/usr/bin/env python3
"""EXP-088 historical extension: chronological research, never live betting ROI.

Default: train from the first complete historical year (2018), test 2020-2026.
The explicit 2020-start sensitivity protocol reuses frozen modern artifacts;
the 2018-start protocol fits every requested year with the same fixed procedure.
Historical closing averages and legacy snapshots do not certify live availability.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_exp081_failures import bootstrap
from scripts.benchmark_model_redesign import align_legacy_context, temporal_blocks
from scripts.benchmark_siamese_architectures import attach_market
from scripts.build_siamese_research_dataset import sha256
from scripts.experiment_betting_calibration import (
    IDENTITIES,
    METADATA,
    MIN_EV,
    TAX,
    evaluate,
    market_calibration_followup,
    plot_reliability,
    policy,
    reliability,
    run_calibration,
)
from scripts.train_and_tune_siamese_series import build_training_features
from src.analysis.probability_metrics import binary_log_loss_vector
from src.models.redesign_series import (
    MODEL_VERSION,
    SeriesEstimator,
    _OddBasis,
    _calibrate,
)
from src.models.symmetric_series import (
    _PROBABILITY_FIELDS,
    _RAW_DIFFERENCES,
    _REQUIRED_BASE_FIELDS,
    _W20_FIELDS,
)

# Recorded in the original benchmark's folds[].models.fixed_ridge.artifact_sha256.
# Do not silently accept regenerated modern weights even if their predictions agree.
FROZEN_SHA256 = {
    2024: "6991068de748cfee2c92281be5bcf84d8c69f54a5d13fa7184bfcd586f5682da",
    2025: "a13b7e9d1dbd7fc7d1d99c1ac39510ebe02b12e7d7ed17a0d7518e6da874e996",
    2026: "22caf6df15453af3d0e3b46b288d8a74fa3e6d682cfb449e83fbe014ab501a2c",
}
BASE_CONFIG = {
    "C": 0.1,
    "fit_intercept": False,
    "solver": "lbfgs",
    "max_iter": 10000,
    "tol": 1e-8,
    "random_state": 83,
}
LIMITATIONS = [
    "Historical averaged closing odds are not independently timestamped paired live quotes.",
    "No live ROI, executable return, profit guarantee, production change, or promotion claim.",
    "Chronological splits do not establish full point-in-time provenance of legacy features, rating states, rosters, or rolling windows.",
    "The registry blending rule is applied to chronological research ridge79, NOT historical operational EXP081; no older stored_exp081 probabilities are fabricated.",
    "2018-start and original 2020-start are different training protocols. Modern weights are reused only for the original protocol; expanded-history probabilities need not equal its predictions.",
    "The first complete historical training year is 2018; the partial November-December 2017 cohort (11 legacy rows) is excluded from training.",
    "Previously inspected historical outcomes and this extension are exploratory, not a new untouched promotion holdout.",
    "Own selections can change both membership and side; fixed calibrated ridge79 selection and its fixed [35%,45%) reference subset are separately reported.",
    "The [35%,45%) bin is not a claim of individual-match or pointwise 40% probability truth.",
    "Monthly block bootstrap uses 5000 resamples; fewer than two months cannot support an interval. Wilson intervals are descriptive IID references only.",
    "Groups below N=20 are descriptive/tiny, omitted from reliability plots, and not independent evidence of success; subgroup comparisons are not multiplicity-adjusted.",
    "The final year may be partial; observed coverage endpoints, not calendar labels, define the available cohort.",
]


def save_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as handle:
        handle.write(payload)


def save_csv(path, frame):
    with Path(path).open("x") as handle:
        frame.to_csv(handle, index=False)


def read_frame(path):
    frame = pd.read_csv(path, dtype=IDENTITIES, low_memory=False)
    if frame.golgg_match_id.isna().any() or frame.golgg_match_id.duplicated().any():
        raise ValueError(f"missing/duplicate match identity: {path}")
    if pd.to_datetime(frame.date, errors="raise").isna().any():
        raise ValueError(f"missing date: {path}")
    return frame.sort_values(["date", "golgg_match_id"]).reset_index(drop=True)


def snapshot_sources(output, inputs):
    # Load the market helper's lazy dependencies before recording source hashes.
    # Importing these modules does not instantiate an inference model.
    import_module("betting_app.core.models.engine")
    import_module("betting_app.core.models.registry")
    sources = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT):
                if path.relative_to(ROOT).parts[0] in ("scripts", "src", "betting_app"):
                    sources.add(path)
    sources.update(
        ROOT / name
        for name in (
            "src/analysis/metrics.py",
            "src/analysis/probability_metrics.py",
            "src/models/symmetric_series.py",
            "scripts/experiment_betting_calibration.py",
        )
    )
    paths = sorted(sources | {path.resolve() for path in inputs})
    hashes = {str(path): sha256(path) for path in paths}
    for source in sorted(sources):
        target = output / "source_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
        if sha256(target) != hashes[str(source)]:
            raise ValueError(f"source changed while snapshotting: {source}")
    save_json(output / "source_manifest_before.json", hashes)
    return hashes


def verify_sources(hashes):
    after = {path: sha256(Path(path)) for path in hashes}
    changed = [path for path in hashes if hashes[path] != after[path]]
    if changed:
        raise ValueError(f"inputs/source dependencies changed during run: {changed}")
    return after


def stages_for(full, year, train_start):
    masks = temporal_blocks(
        full.date, year, protocol="semester-calibration", train_start=train_start
    )
    bounds = {
        "train": (train_start, f"{year - 1}-01-01"),
        "stop": (f"{year - 1}-01-01", f"{year - 1}-04-01"),
        "select": (f"{year - 1}-04-01", f"{year - 1}-07-01"),
        "calibration": (f"{year - 1}-07-01", f"{year}-01-01"),
        "test": (f"{year}-01-01", f"{year + 1}-01-01"),
    }
    dates = pd.to_datetime(full.date)
    claimed = np.zeros(len(full), dtype=bool)
    previous_max = None
    records = {}
    for stage, (start, end) in bounds.items():
        mask = masks[stage]
        np.testing.assert_array_equal(
            mask, ((dates >= start) & (dates < end)).to_numpy()
        )
        if not mask.any() or np.any(claimed & mask):
            raise ValueError(
                f"{year}/{stage}: empty or overlapping chronological stage"
            )
        observed_min, observed_max = dates[mask].min(), dates[mask].max()
        if previous_max is not None and observed_min <= previous_max:
            raise ValueError(f"{year}/{stage}: stages are not strictly chronological")
        claimed |= mask
        previous_max = observed_max
        records[stage] = {
            "start_inclusive": start,
            "end_exclusive": end,
            "n": int(mask.sum()),
            "min": str(full.loc[mask, "date"].min()),
            "max": str(full.loc[mask, "date"].max()),
            "market_n": int(full.loc[mask, "market"].notna().sum()),
        }
    return masks, records


def feature_matrices(full):
    x79, canonical_names = build_training_features(full)
    if x79.shape[1] != 79 or len(set(canonical_names)) != 79:
        raise ValueError("canonical79 feature contract differs")
    names = tuple(["d_" + name for name in canonical_names] + ["c_bo3", "c_bo5"])
    context = np.column_stack(
        ((full.best_of == 3).astype(float), (full.best_of == 5).astype(float))
    )
    x = np.column_stack((x79, context))
    # Rebuild the canonical matrix from actual reversed raw inputs, not from 1-p.
    raw_columns = sorted(_REQUIRED_BASE_FIELDS | {"best_of"} | ({"BoN"} & set(full)))
    reversed_raw = full[raw_columns].copy()
    for column in _PROBABILITY_FIELDS:
        reversed_raw[column] = 1 - full[column]
    for _, left, right in _RAW_DIFFERENCES:
        reversed_raw[left], reversed_raw[right] = full[right], full[left]
    for field in _W20_FIELDS:
        left, right = f"t1_rolling_{field}", f"t2_rolling_{field}"
        reversed_raw[left], reversed_raw[right] = full[right], full[left]
    reversed79, reversed_names = build_training_features(reversed_raw)
    if reversed_names != canonical_names:
        raise ValueError("reversed canonical feature names differ")
    np.testing.assert_allclose(reversed79, -x79, atol=1e-10, rtol=0)
    return x, np.column_stack((reversed79, context)), names


def fit_fixed_ridge(full, x, names, masks, stages, year):
    basis = _OddBasis(79, names, context_indices=())
    transformed = basis.fit_transform(x[masks["train"]], use_splines=False)
    linear = LogisticRegression(**BASE_CONFIG)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        linear.fit(transformed, full.loc[masks["train"], "y_true"].to_numpy(float))
    model = SeriesEstimator(
        feature_names=names,
        n_odd=79,
        kind="linear",
        base=linear,
        basis=basis,
        metadata={
            "model_version": MODEL_VERSION,
            "candidate": "fixed_ridge",
            "year": year,
            "feature_names": list(names),
            "n_odd": 79,
            "schema": {
                "odd": list(names[:79]),
                "even": list(names[79:]),
                "swap": "(-d,c)",
            },
            "seed": 83,
            "refit": False,
            "base_selection_stage": "fixed_predeclared",
            "preprocessing_fit_stage": "train",
            "stage_rows": {stage: record["n"] for stage, record in stages.items()},
            "stages": stages,
            "basis": "odd_linear",
            "center": False,
            "even_context_interactions": [],
            "spline": None,
            "scales": basis.scaler.scale_.tolist(),
            "logistic_regression": BASE_CONFIG,
            "optimizer_iterations": int(linear.n_iter_[0]),
            "stop_and_select": "preserved disjoint stages; no hyperparameter selection for fixed_ridge",
        },
    )
    model.metadata["stop_log_loss"] = float(
        binary_log_loss_vector(
            full.loc[masks["stop"], "y_true"], model.raw_probability(x[masks["stop"]])
        ).mean()
    )
    _calibrate(
        model,
        x[masks["calibration"]],
        full.loc[masks["calibration"], "y_true"].to_numpy(float),
    )
    return model


def assert_identity(actual, expected, columns):
    for column in columns:
        np.testing.assert_array_equal(
            actual[column].to_numpy(), expected[column].to_numpy(), err_msg=column
        )


def build_reference(
    full, original, frozen_summary, benchmark, output, years, train_start
):
    x, reversed_x, names = feature_matrices(full)
    if tuple(frozen_summary["feature_names"]) != names:
        raise ValueError("frozen benchmark canonical feature names differ")
    original_folds = {fold["year"]: fold for fold in frozen_summary["folds"]}
    destination = output / "benchmark" / "legacy79"
    destination.mkdir(parents=True, exist_ok=False)
    rows, folds, stage_rows = [], [], []
    identity_columns = METADATA + ["team1_id", "team2_id", "team1_name", "team2_name"]
    reference_columns = identity_columns + ["odds_a", "odds_b", "market"]
    for year in years:
        masks, stages = stages_for(full, year, train_start)
        target = destination / f"{year}-fixed_ridge.joblib"
        reuse_frozen = train_start == "2020-01-01" and year in FROZEN_SHA256
        if reuse_frozen:
            frozen = original_folds[year]
            expected_hash = FROZEN_SHA256[year]
            if frozen["models"]["fixed_ridge"]["artifact_sha256"] != expected_hash:
                raise ValueError(
                    f"{year}: saved benchmark hash differs from pinned reference"
                )
            for stage in stages:
                for key in ("n", "min", "max"):
                    if stages[stage][key] != frozen["stages"][stage][key]:
                        raise ValueError(
                            f"{year}/{stage}/{key}: frozen chronology differs"
                        )
            source = benchmark / "legacy79" / target.name
            if sha256(source) != expected_hash:
                raise ValueError(f"{year}: original fixed_ridge artifact hash differs")
            with target.open("xb") as handle:
                handle.write(source.read_bytes())
            if sha256(target) != expected_hash:
                raise ValueError(f"{year}: byte-identical artifact copy failed")
            model = SeriesEstimator.load(source)
            provenance = "byte-identical frozen modern fixed_ridge; no refit"
        else:
            model = fit_fixed_ridge(full, x, names, masks, stages, year)
            model.save(target)
            provenance = f"fresh chronological research fixed_ridge; training starts {train_start}; no modern weights backcast"
        if model.feature_names != names or model.n_odd != 79 or model.kind != "linear":
            raise ValueError(
                f"{year}: saved model feature/architecture contract differs"
            )
        if model.basis.context_indices != () or model.basis.spline is not None:
            raise ValueError(
                f"{year}: model is not the frozen odd-linear fixed_ridge procedure"
            )
        for key, value in BASE_CONFIG.items():
            if model.base.get_params()[key] != value:
                raise ValueError(f"{year}: frozen base configuration differs at {key}")
        raw, calibrated = model.raw_probability(x), model.predict(x)
        loaded = SeriesEstimator.load(target)
        # Entire aligned input, including every stage, not a ten-row serialization probe.
        np.testing.assert_array_equal(loaded.raw_probability(x), raw)
        np.testing.assert_array_equal(loaded.predict(x), calibrated)
        reverse_raw, reverse_p = (
            loaded.raw_probability(reversed_x),
            loaded.predict(reversed_x),
        )
        np.testing.assert_allclose(raw + reverse_raw, 1, atol=1e-12, rtol=0)
        np.testing.assert_allclose(calibrated + reverse_p, 1, atol=1e-12, rtol=0)
        chosen, side = policy(calibrated, full.odds_a, full.odds_b)
        reverse_chosen, reverse_side = policy(reverse_p, full.odds_b, full.odds_a)
        np.testing.assert_array_equal(chosen, reverse_chosen)
        np.testing.assert_array_equal(side[chosen], ~reverse_side[chosen])
        part = full.loc[masks["test"], reference_columns].copy().reset_index(drop=True)
        part["fixed_ridge"] = calibrated[masks["test"]]
        part["fixed_ridge__raw"] = raw[masks["test"]]
        part["test_year"] = year
        part["model_sha256"] = sha256(target)
        modern_check = None
        if year in FROZEN_SHA256:
            expected = original.loc[original.date.str[:4] == str(year)].reset_index(
                drop=True
            )
            assert_identity(part, expected, identity_columns)
            modern_check = {"strict_probability_replay": reuse_frozen}
            for column in (
                "fixed_ridge",
                "fixed_ridge__raw",
                "odds_a",
                "odds_b",
                "market",
            ):
                if reuse_frozen or column in ("odds_a", "odds_b", "market"):
                    np.testing.assert_allclose(
                        part[column],
                        expected[column],
                        atol=1e-12,
                        rtol=0,
                        equal_nan=True,
                        err_msg=column,
                    )
                valid = part[column].notna().to_numpy()
                modern_check[column] = (
                    float(
                        np.max(
                            np.abs(
                                part.loc[valid, column].to_numpy()
                                - expected.loc[valid, column].to_numpy()
                            )
                        )
                    )
                    if valid.any()
                    else None
                )
        for stage, mask in masks.items():
            stage_part = full.loc[mask, reference_columns].copy()
            stage_part["test_year"] = year
            stage_part["stage"] = stage
            stage_part["fixed_ridge__raw"] = raw[mask]
            stage_part["fixed_ridge"] = calibrated[mask]
            stage_rows.append(stage_part)
        rows.append(part)
        folds.append(
            {
                "year": year,
                "stages": stages,
                "artifact": str(target),
                "artifact_sha256": sha256(target),
                "provenance": provenance,
                "metadata": model.metadata,
                "full_saved_model_replay_rows": len(full),
                "raw_swapped_input_symmetry_max_error": float(
                    np.max(np.abs(raw + reverse_raw - 1))
                ),
                "calibrated_swapped_input_symmetry_max_error": float(
                    np.max(np.abs(calibrated + reverse_p - 1))
                ),
                "original_reference_max_errors": modern_check,
            }
        )
        print(
            json.dumps(
                {
                    "year": year,
                    "test_n": len(part),
                    "market_n": int(part.market.notna().sum()),
                    "provenance": provenance,
                },
                allow_nan=False,
            ),
            flush=True,
        )
    reference = pd.concat(rows, ignore_index=True)
    save_csv(destination / "predictions.csv", reference)
    save_csv(
        destination / "stage_predictions.csv", pd.concat(stage_rows, ignore_index=True)
    )
    save_json(
        destination / "summary.json",
        {"feature_names": list(names), "n_odd": 79, "folds": folds},
    )
    return reference, folds


def cohort_masks(frame, years):
    actual_years = pd.to_datetime(frame.date).dt.year.to_numpy()
    return {
        "overall": np.ones(len(frame), bool),
        "early_before_2024": actual_years < 2024,
        "modern_2024_plus": actual_years >= 2024,
        **{f"year_{year}": actual_years == year for year in years},
    }


def annotate_reliability(record):
    record["support"] = (
        "no observations"
        if record["n"] == 0
        else (
            "tiny; descriptive only (N<20)"
            if record["n"] < 20
            else "descriptive retrospective group"
        )
    )
    if record["n"] and record["signed_error_monthly_ci"] is None:
        record["monthly_ci_status"] = "unavailable: fewer than two observed months"


def summarize_cohorts(frame, output, prefix, years, overall_results=None):
    names = [name for name in frame if name.startswith("p__")]
    cohorts, table, group_table = {}, [], []
    for cohort, mask in cohort_masks(frame, years).items():
        part = frame.loc[mask].reset_index(drop=True)
        if part.empty:
            cohorts[cohort] = {"n": 0, "status": "no eligible observations"}
            continue
        if cohort == "overall" and overall_results is not None:
            results = overall_results
        else:
            results, _ = evaluate(part, names)
        cohorts[cohort] = {
            "n": len(part),
            "market_n": int(part.market.notna().sum()),
            "models": results,
        }
        y, baseline = part.y_true.to_numpy(), part.p__baseline79.to_numpy()
        for name, result in results.items():
            for selection in ("own_selection", "fixed_baseline_selection"):
                for record in result[selection].values():
                    annotate_reliability(record)
            annotate_reliability(result["fixed_reference_40"])
            for selection, record in (
                ("own", result["own_selection"]["all_selected"]),
                ("fixed_baseline", result["fixed_baseline_selection"]["all_selected"]),
                ("own_p_35_45", result["own_selection"]["p_0.35_0.45"]),
                (
                    "fixed_selection_candidate_p_35_45",
                    result["fixed_baseline_selection"]["p_0.35_0.45"],
                ),
                ("fixed_reference_p_35_45", result["fixed_reference_40"]),
            ):
                ci = record.get("signed_error_monthly_ci") or {}
                table.append(
                    {
                        "family": prefix,
                        "cohort": cohort,
                        "model": name,
                        "all_n": len(part),
                        "market_n": int(part.market.notna().sum()),
                        "overall_log_loss": result["overall"]["log_loss"],
                        "overall_brier": result["overall"]["brier"],
                        "selection": selection,
                        "selected_n": record["n"],
                        "mean_probability": record.get("mean_probability"),
                        "observed_win_fraction": record.get("observed_win_fraction"),
                        "signed_error": record.get(
                            "signed_error_probability_minus_result"
                        ),
                        "ci95_low": ci.get("ci95_low"),
                        "ci95_high": ci.get("ci95_high"),
                        "months": ci.get("months"),
                        "resamples": ci.get("resamples"),
                        "selected_log_loss": record.get("log_loss"),
                        "selected_brier": record.get("brier"),
                        "support": record["support"],
                    }
                )
            # Proper losses remain meaningful for tiny/single-class groups; do not
            # fit a fragile posthoc subgroup logistic slope merely to populate a cell.
            p = part[name].to_numpy()
            grouped = {}
            for dimension in ("best_of", "competition_tier"):
                for value in sorted(part[dimension].unique()):
                    group = part[dimension].to_numpy() == value
                    record = reliability(y[group], p[group], part.loc[group, "date"])
                    annotate_reliability(record)
                    record["paired_LL_vs_baseline"] = bootstrap(
                        binary_log_loss_vector(y[group], p[group])
                        - binary_log_loss_vector(y[group], baseline[group]),
                        part.loc[group, "date"],
                    )
                    record["paired_Brier_vs_baseline"] = bootstrap(
                        (p[group] - y[group]) ** 2 - (baseline[group] - y[group]) ** 2,
                        part.loc[group, "date"],
                    )
                    grouped[f"{dimension}:{value}"] = record
                    group_table.append(
                        {
                            "family": prefix,
                            "cohort": cohort,
                            "model": name,
                            "dimension": dimension,
                            "value": value,
                            "n": record["n"],
                            "log_loss": record["log_loss"],
                            "brier": record["brier"],
                            "support": record["support"],
                        }
                    )
            result["global_proper_metrics_by_format_tier"] = grouped
        plot_reliability(results, output, prefix=f"{prefix}_{cohort}")
    save_json(output / f"{prefix}_cohort_comparisons.json", cohorts)
    save_csv(output / f"{prefix}_comparison.csv", pd.DataFrame(table))
    save_csv(output / f"{prefix}_global_format_tier.csv", pd.DataFrame(group_table))
    return cohorts, table


def check_prior_calibration(aligned, prior, years, train_start):
    modern = aligned.loc[pd.to_datetime(aligned.date).dt.year >= 2024].reset_index(
        drop=True
    )
    expected = prior.loc[pd.to_datetime(prior.date).dt.year.isin(years)].reset_index(
        drop=True
    )
    assert_identity(modern, expected, METADATA)
    columns = [
        "p__baseline79",
        "raw__baseline79",
        "p__format_tier_temperature",
        "p__selected_weighted_temperature",
        "p__symmetric_isotonic",
    ]
    errors = {}
    for column in columns:
        if train_start == "2020-01-01":
            np.testing.assert_allclose(
                modern[column], expected[column], atol=1e-12, rtol=0, err_msg=column
            )
        errors[column] = (
            float(
                np.max(np.abs(modern[column].to_numpy() - expected[column].to_numpy()))
            )
            if len(modern)
            else None
        )
    return {
        "n": len(modern),
        "strict_probability_replay": train_start == "2020-01-01",
        "tolerance": 1e-12 if train_start == "2020-01-01" else None,
        "max_absolute_differences": errors,
        "interpretation": "unchanged original protocol"
        if train_start == "2020-01-01"
        else "different train start; differences are expected, not replay failures",
    }


def run(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    created_at = datetime.now(timezone.utc).isoformat()
    reference_path = args.benchmark_dir / "legacy79" / "predictions.csv"
    frozen_summary_path = args.benchmark_dir / "legacy79" / "summary.json"
    prior_path = args.previous_run / "predictions.csv"
    inputs = [
        args.legacy,
        args.replay,
        args.odds,
        reference_path,
        frozen_summary_path,
        prior_path,
        args.previous_run / "summary.json",
    ]
    inputs.extend(
        args.benchmark_dir / "legacy79" / f"{year}-fixed_ridge.joblib"
        for year in FROZEN_SHA256
    )
    hashes = snapshot_sources(output, inputs)
    try:
        versions = {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "matplotlib",
                "lightgbm",
            )
        }
        save_json(
            output / "run_config.json",
            {
                "experiment": "EXP-088-historical-extension",
                "created_at": created_at,
                "command": [sys.executable, *sys.argv],
                "years": list(args.years),
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "packages": versions,
                },
                "protocol": "semester-calibration",
                "train_start": args.train_start,
                "modern_artifact_reuse": args.train_start == "2020-01-01",
                "base_model": BASE_CONFIG,
                "frozen_modern_sha256": FROZEN_SHA256,
                "monthly_bootstrap_resamples": 5000,
                "policy": {
                    "tax": TAX,
                    "min_ev_strictly_greater": MIN_EV,
                    "one_side_per_series": True,
                    "tie": "no selection",
                },
                "limitations": LIMITATIONS,
            },
        )
        legacy, replay = read_frame(args.legacy), read_frame(args.replay)
        full, join_audit = align_legacy_context(legacy, replay)
        odds = pd.read_csv(
            args.odds, dtype={"golgg_match_id": "string"}, low_memory=False
        )
        full, market_audit = attach_market(full, odds)
        full = full.sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
        if (
            full.golgg_match_id.duplicated().any()
            or not np.isin(full.y_true, (0, 1)).all()
        ):
            raise ValueError(
                "aligned data violates unique-series/binary-label contract"
            )
        original, prior = read_frame(reference_path), read_frame(prior_path)
        frozen_summary = json.loads(frozen_summary_path.read_text())
        reference, folds = build_reference(
            full,
            original,
            frozen_summary,
            args.benchmark_dir,
            output,
            args.years,
            args.train_start,
        )
        predictions, calibrators = run_calibration(
            full,
            reference,
            output / "benchmark",
            output,
            years=args.years,
            train_start=args.train_start,
        )
        aligned = predictions.merge(
            reference[
                [
                    "golgg_match_id",
                    "team1_id",
                    "team2_id",
                    "team1_name",
                    "team2_name",
                    "odds_a",
                    "odds_b",
                    "market",
                    "test_year",
                    "model_sha256",
                ]
            ],
            on="golgg_match_id",
            validate="one_to_one",
            sort=False,
        )
        # An actual raw-before-H2 variant makes before/after probabilities and selected
        # N observable under exactly the same tax/EV rule as the calibrated variants.
        aligned["p__raw_baseline79"] = aligned.raw__baseline79
        modern_replay = check_prior_calibration(
            aligned, prior, args.years, args.train_start
        )
        save_csv(output / "predictions.csv", aligned)
        results, selections = evaluate(
            aligned, [name for name in aligned if name.startswith("p__")]
        )
        save_csv(output / "selected_diagnostic_rows.csv", selections)
        sports_cohorts, sports_table = summarize_cohorts(
            aligned, output, "sports", args.years, results
        )
        market_summary = market_calibration_followup(
            full,
            reference,
            output / "benchmark",
            output,
            years=args.years,
            train_start=args.train_start,
            registered_sports_column="fixed_ridge",
        )
        market_frame = read_frame(output / "market_conditioned_predictions.csv")
        market_cohorts, market_table = summarize_cohorts(
            market_frame, output, "market", args.years, market_summary["models"]
        )
        save_csv(output / "comparison.csv", pd.DataFrame(sports_table + market_table))
        coverage_rows = []
        for cohort, mask in cohort_masks(aligned, args.years).items():
            part = aligned.loc[mask]
            base_mask, _ = policy(part.p__baseline79, part.odds_a, part.odds_b)
            raw_mask, _ = policy(part.p__raw_baseline79, part.odds_a, part.odds_b)
            coverage_rows.append(
                {
                    "cohort": cohort,
                    "rows": len(part),
                    "market_rows": int(part.market.notna().sum()),
                    "raw_selected_n": int(raw_mask.sum()),
                    "fixed_ridge_selected_n": int(base_mask.sum()),
                    "min_date": None if part.empty else str(part.date.min()),
                    "max_date": None if part.empty else str(part.date.max()),
                }
            )
        save_csv(output / "coverage.csv", pd.DataFrame(coverage_rows))
        before = prior.loc[pd.to_datetime(prior.date).dt.year.isin(args.years)]
        before_market = original.loc[
            pd.to_datetime(original.date).dt.year.isin(args.years), "market"
        ].notna()
        summary = {
            "experiment": "EXP-088-historical-extension",
            "status": "completed",
            "created_at": created_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "years": list(args.years),
            "rows": len(aligned),
            "market_rows": int(aligned.market.notna().sum()),
            "train_start": args.train_start,
            "protocol_difference": "original 2020-start; modern artifacts reused"
            if args.train_start == "2020-01-01"
            else "extended 2018-start; every yearly ridge fitted afresh, including modern years",
            "before_vs_after_coverage": {
                "prior_modern_n": len(before),
                "prior_modern_market_n": int(before_market.sum()),
                "expanded_n": len(aligned),
                "expanded_market_n": int(aligned.market.notna().sum()),
                "added_early_n": int(
                    (pd.to_datetime(aligned.date).dt.year < 2024).sum()
                ),
                "cohorts": coverage_rows,
            },
            "modern_calibration_replay_against_previous_run": modern_replay,
            "join_audit": join_audit,
            "market_audit": market_audit,
            "folds": folds,
            "calibrators": calibrators,
            "sports_cohorts": sports_cohorts,
            "market_cohorts": market_cohorts,
            "market_conditioned_followup": market_summary,
            "policy": {
                "tax": TAX,
                "min_ev_strictly_greater": MIN_EV,
                "one_side_per_series": True,
                "executable_rows_with_complete_timestamp_chain": 0,
            },
            "promotion": {"approved": False, "production_changes": False},
            "limitations": LIMITATIONS,
            "source_sha256": hashes,
            "runtime_seconds": time.monotonic() - started,
        }
        save_json(output / "source_manifest_after.json", verify_sources(hashes))
        save_json(output / "summary.json", summary)
        save_json(
            output / "output_sha256.json",
            {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "coverage": coverage_rows,
                    "runtime_seconds": summary["runtime_seconds"],
                    "promotion": False,
                },
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )
    except Exception as error:
        # Preserve failed runs and every artifact already created. Never rerun in-place.
        integrity_error = None
        try:
            verify_sources(hashes)
        except Exception as integrity:
            integrity_error = str(integrity)
        save_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "source_integrity_error": integrity_error,
                "runtime_seconds": time.monotonic() - started,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=ROOT / "data/artifacts/model-redesign-v1/benchmark-semester",
    )
    parser.add_argument(
        "--previous-run",
        type=Path,
        default=ROOT
        / "data/artifacts/exp088-betting-calibration/run03-all-experiments",
    )
    parser.add_argument(
        "--train-start", choices=("2018-01-01", "2020-01-01"), default="2018-01-01"
    )
    parser.add_argument("--first-test-year", type=int, default=2020)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    earliest = int(args.train_start[:4]) + 2
    if (
        args.first_test_year < earliest
        or args.last_test_year > 2026
        or args.first_test_year > args.last_test_year
    ):
        parser.error(
            f"test years must be an ascending nonempty range within {earliest}..2026 for this train start"
        )
    args.years = tuple(range(args.first_test_year, args.last_test_year + 1))
    run(args)


if __name__ == "__main__":
    main()
