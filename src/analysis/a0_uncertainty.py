"""Development-only evaluation for the frozen Causal A0 uncertainty study.

This module complements (and never replaces) the canonical research benchmark.
All calibration cutoffs are learned from each annual CAL partition; TEST labels
are used only after selections are frozen.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.metrics import calculate_ece
from src.analysis.research_benchmark import IDENTITY, losses, paired_delta, validate_frame


YEARS = (2024, 2025, 2026)
VARIANTS = ("seed_only", "block_bootstrap")
RETENTION_LEVELS = (1.0, 0.9, 0.75, 0.5)
FAVORITE_PROBABILITY_BANDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BOOTSTRAP_DRAWS_MIN = 5000
CANONICAL_EXPECTED_ROWS = 11550
EXPECTED_PARTITION_ROWS = {
    2024: {"cal": 4836, "test": 4642},
    2025: {"cal": 4642, "test": 3805},
    2026: {"cal": 3805, "test": 3103},
}
LL_NONINFERIORITY_MARGIN = 0.002
BOOTSTRAP_SEED = 20260915
MEMBER_COLUMNS = tuple(f"p_member_{index}" for index in range(5))
OPTIONAL_MODES = {"no_w20": "p_no_w20", "no_organization": "p_no_organization"}
PROFILE_COLUMNS = {
    "year": "model_year",
    "best_of": "best_of",
    "experience": "experience",
    "unknown_players": "unknown_players",
    "new_organization": "new_organization",
    "w20_min_gate": "w20_min_gate",
}


def _as_probability_array(values: pd.Series, name: str) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all() or ((array <= 0.0) | (array >= 1.0)).any():
        raise ValueError(f"invalid probability in {name}; expected finite 0<p<1")
    return array


def _normalize_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "variant", "model_year", "partition", "train_end", "calibration_end", "p",
        *MEMBER_COLUMNS,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    frame = predictions.copy()
    if frame[["variant", "model_year", "partition"]].isna().any().any():
        raise ValueError("null variant, model_year, or partition")
    frame["variant"] = frame["variant"].astype(str)
    frame["partition"] = frame["partition"].astype(str).str.lower()
    try:
        numeric_year = pd.to_numeric(frame["model_year"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("invalid model_year") from error
    year_values = numeric_year.to_numpy(dtype=float)
    if not np.isfinite(year_values).all() or not np.equal(year_values, np.floor(year_values)).all():
        raise ValueError("invalid model_year")
    frame["model_year"] = numeric_year.astype(int)
    if not set(frame["variant"]).issubset(VARIANTS) or set(frame["variant"]) != set(VARIANTS):
        raise ValueError(f"expected exactly the fixed variants {VARIANTS}")
    if not set(frame["model_year"]).issubset(YEARS) or set(frame["model_year"]) != set(YEARS):
        raise ValueError(f"expected all model years {YEARS}")
    if not set(frame["partition"]).issubset({"cal", "test"}) or set(frame["partition"]) != {"cal", "test"}:
        raise ValueError("expected separate cal and test partitions")
    if frame["golgg_match_id"].isna().any():
        raise ValueError("null match identity")
    frame["golgg_match_id"] = frame["golgg_match_id"].astype(str)
    member_values = frame.loc[:, list(MEMBER_COLUMNS)].to_numpy(dtype=float)
    frame["p_std"] = np.std(member_values, axis=1, ddof=0)
    member_mean = member_values.mean(axis=1)
    _as_probability_array(frame["p"], "p")
    if not np.allclose(member_mean, frame["p"].to_numpy(dtype=float), rtol=1e-7, atol=1e-10):
        raise ValueError("p must equal the arithmetic mean of the five final member probabilities")
    for column in MEMBER_COLUMNS:
        _as_probability_array(frame[column], column)
    if "p_std" in predictions:
        submitted_std = predictions["p_std"].to_numpy(dtype=float)
        if not np.isfinite(submitted_std).all() or not np.allclose(
            submitted_std, frame["p_std"].to_numpy(), rtol=1e-7, atol=1e-10
        ):
            raise ValueError("p_std does not match population spread across the five members")
    frame["entropy"] = _binary_entropy(frame["p"].to_numpy(dtype=float))

    for column in OPTIONAL_MODES.values():
        if column in frame:
            present = frame[column].notna()
            if present.any():
                _as_probability_array(frame.loc[present, column], column)
    duplicate_keys = ["variant", "model_year", "partition", "golgg_match_id"]
    if frame.duplicated(duplicate_keys).any():
        raise ValueError("duplicate candidate identity within variant/year/partition")

    probability_columns = ["p", *MEMBER_COLUMNS]
    nullable = []
    for column in OPTIONAL_MODES.values():
        if column in frame:
            probability_columns.append(column)
            nullable.append(column)
    for (variant, year, partition), group in frame.groupby(
        ["variant", "model_year", "partition"], sort=False
    ):
        validate_frame(group, probability_columns, nullable=nullable)
        _validate_release_bounds(group, model_year=int(year), partition=partition)
        expected_calendar_year = int(year) if partition == "test" else int(year) - 1
        actual_years = pd.to_datetime(group["date"], errors="raise").dt.year
        if not actual_years.eq(expected_calendar_year).all():
            raise ValueError(f"{variant}/{year}/{partition}: date outside the frozen annual partition")
    return frame.reset_index(drop=True)


def _validate_release_bounds(frame: pd.DataFrame, *, model_year: int, partition: str) -> None:
    for column in ("train_end", "calibration_end"):
        if column not in frame or frame[column].isna().any():
            raise ValueError(f"candidate requires {column}")
        values = frame[column].astype(str)
        if not values.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError(f"{column}: calendar-day release bound required")
        pd.to_datetime(values, errors="raise")
        # CAL distributions fit the policy after annual calibration; they are not
        # out-of-sample forecasts. Both fitting stages still precede TEST origin.
        boundary = f"{model_year - 1}-01-01" if column == "train_end" else f"{model_year}-01-01"
        if not (values.to_numpy() < boundary).all():
            raise ValueError(f"future {column}")
    if not (frame.train_end.astype(str) <= frame.calibration_end.astype(str)).all():
        raise ValueError("train_end follows calibration_end")
    if partition == "cal":
        if not (frame.result_day.astype(str) < f"{model_year}-01-01").all():
            raise ValueError("CAL labels must be released before the prediction-year origin")
        if not (frame.result_day.astype(str) <= frame.calibration_end.astype(str)).all():
            raise ValueError("CAL label release follows calibration_end")


def _normalize_baseline(baseline: pd.DataFrame) -> pd.DataFrame:
    frame = baseline.copy()
    validate_frame(frame, ["p_full"])
    frame["golgg_match_id"] = frame["golgg_match_id"].astype(str)
    if frame["golgg_match_id"].duplicated().any():
        raise ValueError("canonical baseline has duplicate match identity")
    _as_probability_array(frame["p_full"], "baseline p_full")
    if len(frame) != CANONICAL_EXPECTED_ROWS:
        raise ValueError(
            f"canonical baseline coverage mismatch: expected {CANONICAL_EXPECTED_ROWS}, got {len(frame)}"
        )
    date_year = pd.to_datetime(frame["date"], errors="raise").dt.year
    frame["model_year"] = date_year.astype(int)
    if set(frame["model_year"]) != set(YEARS):
        raise ValueError("canonical baseline must cover each frozen model year")
    counts = frame.groupby("model_year").size().to_dict()
    for year in YEARS:
        expected = EXPECTED_PARTITION_ROWS[year]["test"]
        if int(counts.get(year, 0)) != expected:
            raise ValueError(f"canonical baseline {year} coverage mismatch")
    return frame.reset_index(drop=True)


def _check_candidate_coverage_and_identity(
    predictions: pd.DataFrame, baseline: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    base = _normalize_baseline(baseline)
    candidate = _normalize_prediction_frame(predictions)
    expected_keys = {
        (variant, year, partition)
        for variant in VARIANTS for year in YEARS for partition in ("cal", "test")
    }
    actual_keys = set(
        candidate.groupby(["variant", "model_year", "partition"], sort=False).groups
    )
    missing_keys = expected_keys - actual_keys
    if missing_keys:
        raise ValueError(f"missing annual variant partitions: {sorted(missing_keys)}")
    baseline_by_year: dict[int, pd.DataFrame] = {}
    for year in YEARS:
        part = base.loc[base["model_year"].eq(year)].copy()
        if len(part) != EXPECTED_PARTITION_ROWS[year]["test"]:
            raise ValueError(f"canonical baseline {year} coverage mismatch")
        baseline_by_year[year] = part

    aligned: dict[tuple[str, int], pd.DataFrame] = {}
    calibration_by_year: dict[int, pd.DataFrame] = {}
    identity_columns = IDENTITY[1:] + ["result_day", "feature_source_max_day"]
    for variant in VARIANTS:
        for year in YEARS:
            for partition in ("cal", "test"):
                count = int(EXPECTED_PARTITION_ROWS[year][partition])
                part = candidate.loc[
                    candidate["variant"].eq(variant)
                    & candidate["model_year"].eq(year)
                    & candidate["partition"].eq(partition)
                ]
                if len(part) != count:
                    raise ValueError(
                        f"{variant}/{year}/{partition} coverage mismatch: expected {count}, got {len(part)}"
                    )
                if partition == "cal" and year not in calibration_by_year:
                    calibration_by_year[year] = part
                    continue
                expected = (
                    calibration_by_year[year] if partition == "cal" else baseline_by_year[year]
                )
                if set(part["golgg_match_id"]) != set(expected["golgg_match_id"]):
                    raise ValueError(f"{variant}/{year}/{partition} coverage does not match fixed cohort")
                expected_by_id = expected.set_index("golgg_match_id")
                ordered_expected = expected_by_id.loc[part["golgg_match_id"]]
                for column in identity_columns:
                    if column in ("team1_id", "team2_id", "date", "result_day", "feature_source_max_day"):
                        matches = part[column].astype(str).to_numpy() == ordered_expected[column].astype(str).to_numpy()
                    else:
                        matches = part[column].to_numpy() == ordered_expected[column].to_numpy()
                    if not bool(np.asarray(matches).all()):
                        raise ValueError(f"{variant}/{year}/{partition} identity mismatch: {column}")
                if partition == "test":
                    aligned[(variant, year)] = part.reset_index(drop=True)
    return base, candidate, aligned


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def _fit_cutoff(cal: pd.DataFrame, score_column: str, retention: float) -> dict[str, Any]:
    scores = cal[score_column].to_numpy(dtype=float)
    ids = cal["golgg_match_id"].astype(str).to_numpy()
    n = len(cal)
    requested = min(n, max(1, int(math.ceil(n * retention))))
    if requested == n:
        return {
            "threshold": None,
            "threshold_method": "retain_all",
            "cal_rows": n,
            "cal_requested_rows": requested,
            "cal_selected_rows": n,
            "cal_coverage": 1.0,
            "cal_rows_below_threshold": None,
            "cal_threshold_ties": 0,
            "cal_tied_rows_selected": 0,
            "cal_tie_fraction_selected": 0.0,
        }
    order = np.lexsort((ids, scores))
    threshold = float(scores[order[requested - 1]])
    below = int(np.sum(scores < threshold))
    tied = int(np.sum(scores == threshold))
    tied_selected = requested - below
    return {
        "threshold": threshold,
        "threshold_method": "CAL empirical quantile; equal-score ties split by ascending match ID",
        "cal_rows": n,
        "cal_requested_rows": requested,
        "cal_selected_rows": requested,
        "cal_coverage": requested / n,
        "cal_rows_below_threshold": below,
        "cal_threshold_ties": tied,
        "cal_tied_rows_selected": tied_selected,
        "cal_tie_fraction_selected": tied_selected / tied,
    }


def _select_from_cutoff(
    frame: pd.DataFrame, score_column: str, cutoff: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(frame)
    threshold = cutoff["threshold"]
    if threshold is None:
        selected = np.ones(n, dtype=bool)
        detail = {
            "test_rows_below_threshold": None,
            "test_threshold_ties": 0,
            "test_tied_rows_selected": 0,
            "tie_break": "all rows retained",
        }
    else:
        scores = frame[score_column].to_numpy(dtype=float)
        ids = frame["golgg_match_id"].astype(str).to_numpy()
        below = scores < threshold
        ties = scores == threshold
        tied_indices = np.flatnonzero(ties)
        tied_requested = int(math.floor(
            cutoff["cal_tie_fraction_selected"] * len(tied_indices) + 0.5
        ))
        tie_order = tied_indices[np.argsort(ids[tied_indices], kind="stable")]
        selected = below.copy()
        selected[tie_order[:tied_requested]] = True
        detail = {
            "test_rows_below_threshold": int(below.sum()),
            "test_threshold_ties": int(ties.sum()),
            "test_tied_rows_selected": tied_requested,
            "tie_break": "CAL tied-selection fraction; ascending match ID within TEST ties",
        }
    detail.update(test_rows=int(n), test_selected_rows=int(selected.sum()),
                  test_achieved_coverage=float(selected.mean()) if n else None)
    return selected, detail


def _score_metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = frame["y"].to_numpy(dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0:
        empty = {
            "sample_size": 0, "log_loss": None, "brier_score": None,
            "tail_ll_ge_2_5": None, "accuracy": None, "auc": None,
            "ece10": None, "mean_predicted_probability": None,
            "observed_win_rate": None, "calibration_gap_observed_minus_predicted": None,
            "status": "zero_coverage",
        }
        return empty, []
    _as_probability_array(pd.Series(p), "scored probability")
    per_match_loss = losses(y, p)
    y_float = y.astype(float)
    summary = {
        "sample_size": int(len(y)),
        "log_loss": float(np.mean(per_match_loss)),
        "brier_score": float(np.mean(np.square(p - y_float))),
        "tail_ll_ge_2_5": int(np.sum(per_match_loss >= 2.5)),
        "accuracy": float(np.mean((p >= 0.5) == y.astype(bool))),
        "auc": None,
        "ece10": float(calculate_ece(y, p, n_bins=10)),
        "mean_predicted_probability": float(np.mean(p)),
        "observed_win_rate": float(np.mean(y)),
        "calibration_gap_observed_minus_predicted": float(np.mean(y_float - p)),
        "status": "scored",
    }
    if len(np.unique(y)) > 1:
        from sklearn.metrics import roc_auc_score
        summary["auc"] = float(roc_auc_score(y, p))
    calibration_rows = []
    boundaries = np.linspace(0.0, 1.0, 11)
    for bin_index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        mask = (p > lower) & (p <= upper)
        calibration_rows.append({
            "calibration_bin": bin_index,
            "probability_lower_exclusive": float(lower),
            "probability_upper_inclusive": float(upper),
            "sample_size": int(mask.sum()),
            "mean_predicted_probability": float(p[mask].mean()) if mask.any() else None,
            "observed_win_rate": float(y[mask].mean()) if mask.any() else None,
            "calibration_gap_observed_minus_predicted": (
                float(y[mask].mean() - p[mask].mean()) if mask.any() else None
            ),
        })
    return summary, calibration_rows


def _add_score_records(
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
    probability_column: str,
    *,
    metadata: dict[str, Any],
    draws: int,
    metrics: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
) -> None:
    candidate_p = frame[probability_column].to_numpy(dtype=float)
    candidate_summary, candidate_bins = _score_metrics(frame, candidate_p)
    metrics.append({**metadata, "model": "candidate", "probability_column": probability_column,
                    **candidate_summary})
    for row in candidate_bins:
        calibration.append({**metadata, "model": "candidate", "probability_column": probability_column, **row})
    baseline_p = baseline["p_full"].to_numpy(dtype=float)
    baseline_summary, baseline_bins = _score_metrics(baseline, baseline_p)
    metrics.append({**metadata, "model": "canonical_A0", "probability_column": "p_full",
                    **baseline_summary})
    for row in baseline_bins:
        calibration.append({**metadata, "model": "canonical_A0", "probability_column": "p_full", **row})
    if frame.empty:
        for measure in ("log_loss", "brier"):
            comparisons.append({
                **metadata, "model": "candidate", "reference": "canonical_A0",
                "measure": measure, "delta_direction": "candidate_minus_canonical_A0",
                "delta": None, "ci95": None, "blocks": 0, "rows": 0,
                "status": "zero_coverage",
            })
        return
    for measure, delta in (
        ("log_loss", losses(frame.y.to_numpy(dtype=int), candidate_p)
         - losses(baseline.y.to_numpy(dtype=int), baseline_p)),
        ("brier", np.square(candidate_p - frame.y.to_numpy(dtype=float))
         - np.square(baseline_p - baseline.y.to_numpy(dtype=float))),
    ):
        paired = paired_delta(frame, delta, draws)
        comparisons.append({
            **metadata,
            "model": "candidate",
            "reference": "canonical_A0",
            "measure": measure,
            "delta_direction": "candidate_minus_canonical_A0",
            **paired,
        })


def _paired_variant_comparisons(
    candidate_by_variant: dict[str, pd.DataFrame], draws: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aligned = {}
    for variant in VARIANTS:
        frame = pd.concat([candidate_by_variant[(variant, year)] for year in YEARS], ignore_index=True)
        frame = frame.set_index("golgg_match_id", drop=False)
        aligned[variant] = frame
    if set(aligned["seed_only"].index) != set(aligned["block_bootstrap"].index):
        raise ValueError("variant TEST cohorts differ after canonical identity validation")
    ids = aligned["seed_only"].index
    seed = aligned["seed_only"]
    bootstrap = aligned["block_bootstrap"].loc[ids]
    if not np.array_equal(seed.y.to_numpy(), bootstrap.y.to_numpy()):
        raise ValueError("variant TEST labels differ")
    rows = []
    result = {}
    for measure, delta in (
        ("log_loss", losses(bootstrap.y.to_numpy(dtype=int), bootstrap.p.to_numpy(dtype=float))
         - losses(seed.y.to_numpy(dtype=int), seed.p.to_numpy(dtype=float))),
        ("brier", np.square(bootstrap.p.to_numpy(dtype=float) - bootstrap.y.to_numpy(dtype=float))
         - np.square(seed.p.to_numpy(dtype=float) - seed.y.to_numpy(dtype=float))),
    ):
        paired = paired_delta(seed, delta, draws)
        result[measure] = paired
        rows.append({
            "model": "block_bootstrap",
            "reference": "seed_only",
            "measure": measure,
            "delta_direction": "block_bootstrap_minus_seed_only",
            "selector": "all_test_rows",
            "retention": 1.0,
            **paired,
        })
    return rows, result


def _favorite_band(probability: np.ndarray) -> np.ndarray:
    favorite = np.maximum(probability, 1.0 - probability)
    edges = np.asarray(FAVORITE_PROBABILITY_BANDS[1:-1], dtype=float)
    return np.searchsorted(edges, favorite, side="right").astype(int)


def _monthly_standardized_gap(
    frame: pd.DataFrame,
    *,
    band_column: str,
    high_column: str,
    excess_column: str,
    strata_columns: list[str],
    supported_strata: list[tuple[Any, ...]],
    draws: int,
) -> dict[str, Any]:
    months = pd.to_datetime(frame.date).dt.to_period("M").astype(str).to_numpy()
    unique_months = np.asarray(sorted(np.unique(months)))
    strata_values = [tuple(row) for row in frame[strata_columns].itertuples(index=False, name=None)]
    high_values = frame[high_column].to_numpy(dtype=bool)
    excess = frame[excess_column].to_numpy(dtype=float)
    row_strata = np.empty(len(strata_values), dtype=object)
    for index, value in enumerate(strata_values):
        row_strata[index] = value
    eligible = set(supported_strata)
    point, point_valid = _standardized_estimate(
        np.ones(len(frame), dtype=float), row_strata, high_values, excess, eligible
    )
    result: dict[str, Any] = {
        "point_estimate": point,
        "blocks": int(len(unique_months)),
        "block_unit": "calendar month (whole month sampled with replacement)",
        "bootstrap_draws": int(draws),
        "valid_draws": 0,
        "invalid_draws": int(draws),
        "invalid_draw_fraction": 1.0,
        "ci95": None,
        "status": "zero_high_low_overlap" if not supported_strata else "single_month_block",
    }
    if not supported_strata or len(unique_months) < 2:
        return result
    month_index = {month: i for i, month in enumerate(unique_months)}
    row_month = np.fromiter((month_index[value] for value in months), dtype=int, count=len(months))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(draws, dtype=float)
    valid_count = 0
    for draw in range(draws):
        picks = rng.integers(0, len(unique_months), size=len(unique_months))
        multiplicity = np.bincount(picks, minlength=len(unique_months)).astype(float)
        row_weights = multiplicity[row_month]
        estimate, valid = _standardized_estimate(
            row_weights, row_strata, high_values, excess, eligible
        )
        if valid:
            samples[valid_count] = estimate
            valid_count += 1
    invalid = draws - valid_count
    result.update(valid_draws=int(valid_count), invalid_draws=int(invalid),
                  invalid_draw_fraction=float(invalid / draws))
    if valid_count == draws:
        result["ci95"] = [float(np.quantile(samples, 0.025)),
                          float(np.quantile(samples, 0.975))]
        result["status"] = "estimated" if point_valid and len(supported_strata) == len(set(strata_values)) else "partial_band_support"
    else:
        result["status"] = "sparse_bootstrap_support"
    return result


def _standardized_estimate(
    row_weights: np.ndarray,
    row_strata: np.ndarray,
    high: np.ndarray,
    excess: np.ndarray,
    eligible: set[tuple[Any, ...]],
) -> tuple[float | None, bool]:
    strata_n: dict[tuple[Any, ...], float] = {}
    gaps: dict[tuple[Any, ...], float] = {}
    for stratum in eligible:
        mask = np.fromiter((value == stratum for value in row_strata), dtype=bool, count=len(row_strata))
        low_mask = mask & ~high
        high_mask = mask & high
        low_n = float(row_weights[low_mask].sum())
        high_n = float(row_weights[high_mask].sum())
        total = low_n + high_n
        if low_n <= 0.0 or high_n <= 0.0 or total <= 0.0:
            return None, False
        low_mean = float(np.dot(row_weights[low_mask], excess[low_mask]) / low_n)
        high_mean = float(np.dot(row_weights[high_mask], excess[high_mask]) / high_n)
        strata_n[stratum] = total
        gaps[stratum] = high_mean - low_mean
    denominator = sum(strata_n.values())
    if denominator <= 0.0:
        return None, False
    estimate = sum(strata_n[key] * gaps[key] for key in eligible) / denominator
    return float(estimate), True


def _signal_tables_and_report(
    candidate_by_variant: dict[tuple[str, int], pd.DataFrame],
    cal_by_variant: dict[tuple[str, int], pd.DataFrame],
    draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {}
    band_records: list[dict[str, Any]] = []
    for variant in VARIANTS:
        test_parts = []
        per_year: dict[int, pd.DataFrame] = {}
        median_map: dict[tuple[int, int], float | None] = {}
        for year in YEARS:
            cal = cal_by_variant[(variant, year)]
            test = candidate_by_variant[(variant, year)].copy()
            cal_bands = _favorite_band(cal.p.to_numpy(dtype=float))
            test_bands = _favorite_band(test.p.to_numpy(dtype=float))
            for band_index in range(len(FAVORITE_PROBABILITY_BANDS) - 1):
                med = (float(np.median(cal.loc[cal_bands == band_index, "p_std"]))
                       if np.any(cal_bands == band_index) else None)
                median_map[(year, band_index)] = med
            test["favorite_band"] = test_bands
            medians = np.asarray([median_map[(year, band)] for band in test_bands], dtype=object)
            calibrated = np.asarray([value is not None for value in medians], dtype=bool)
            numeric_medians = np.asarray([value if value is not None else np.nan for value in medians], dtype=float)
            test["cal_median_p_std"] = numeric_medians
            test["signal_calibrated"] = calibrated
            test["dispersion_high"] = calibrated & (test.p_std.to_numpy(dtype=float) > numeric_medians)
            test["excess_log_loss"] = (
                losses(test.y.to_numpy(dtype=int), test.p.to_numpy(dtype=float))
                - _binary_entropy(test.p.to_numpy(dtype=float))
            )
            per_year[year] = test
            test_parts.append(test)
            for band_index in range(len(FAVORITE_PROBABILITY_BANDS) - 1):
                all_band_rows = test.loc[test.favorite_band.eq(band_index)]
                med = median_map[(year, band_index)]
                band_rows = all_band_rows.loc[all_band_rows.signal_calibrated]
                high_mask = band_rows.dispersion_high.to_numpy(dtype=bool)
                low_mask = ~high_mask
                high_excess = band_rows.loc[high_mask, "excess_log_loss"].to_numpy(dtype=float)
                low_excess = band_rows.loc[low_mask, "excess_log_loss"].to_numpy(dtype=float)
                n_high, n_low = len(high_excess), len(low_excess)
                band_records.append({
                    "variant": variant,
                    "model_year": year,
                    "favorite_band": _band_label(band_index),
                    "cal_median_p_std": med,
                    "cal_rows_in_band": int(np.sum(cal_bands == band_index)),
                    "test_rows_in_band": int(len(all_band_rows)),
                    "test_rows_with_cal_threshold": int(len(band_rows)),
                    "test_high_dispersion_rows": int(n_high),
                    "test_low_dispersion_rows": int(n_low),
                    "test_median_ties_assigned_low": int(np.sum(band_rows.p_std.to_numpy(dtype=float) == med)) if med is not None else 0,
                    "mean_excess_log_loss_high": float(high_excess.mean()) if n_high else None,
                    "mean_excess_log_loss_low": float(low_excess.mean()) if n_low else None,
                    "raw_excess_gap_high_minus_low": float(high_excess.mean() - low_excess.mean()) if n_high and n_low else None,
                    "status": (
                        "missing_CAL_band_threshold" if med is None
                        else ("estimated" if n_high and n_low else "zero_high_or_low_coverage")
                    ),
                    "p_std_median_tie_policy": "p_std > CAL median is high; equality belongs to low",
                })
        pooled = pd.concat(test_parts, ignore_index=True)
        pooled_strata = ["model_year", "favorite_band"]
        support = _supported_strata(pooled, pooled_strata)
        pooled_result = _monthly_standardized_gap(
            pooled, band_column="favorite_band", high_column="dispersion_high",
            excess_column="excess_log_loss", strata_columns=pooled_strata,
            supported_strata=support, draws=draws,
        )
        pooled_result["strata"] = pooled_strata
        pooled_result["eligible_strata"] = len(support)
        pooled_result["expected_strata"] = len(YEARS) * (len(FAVORITE_PROBABILITY_BANDS) - 1)
        pooled_result["unsupported_strata"] = pooled_result["expected_strata"] - len(support)
        if len(support) != pooled_result["expected_strata"] and pooled_result["status"] == "estimated":
            pooled_result["status"] = "partial_band_support"
        pooled_result["raw_unadjusted_excess_gap"] = _raw_excess_gap(pooled)
        per_year_results = {}
        for year, part in per_year.items():
            support_year = _supported_strata(part, ["favorite_band"])
            year_result = _monthly_standardized_gap(
                part, band_column="favorite_band", high_column="dispersion_high",
                excess_column="excess_log_loss", strata_columns=["favorite_band"],
                supported_strata=support_year, draws=draws,
            )
            year_result["eligible_bands"] = len(support_year)
            year_result["expected_bands"] = len(FAVORITE_PROBABILITY_BANDS) - 1
            year_result["unsupported_bands"] = year_result["expected_bands"] - len(support_year)
            if len(support_year) != year_result["expected_bands"] and year_result["status"] == "estimated":
                year_result["status"] = "partial_band_support"
            year_result["raw_unadjusted_excess_gap"] = _raw_excess_gap(part)
            per_year_results[str(year)] = year_result
        report[variant] = {"overall": pooled_result, "by_model_year": per_year_results}
    return report, band_records


def _supported_strata(frame: pd.DataFrame, columns: list[str]) -> list[tuple[Any, ...]]:
    supported = []
    keys = [tuple(row) for row in frame[columns].itertuples(index=False, name=None)]
    for key in sorted(set(keys), key=lambda item: tuple(str(value) for value in item)):
        mask = np.fromiter((value == key for value in keys), dtype=bool, count=len(keys))
        high = frame.dispersion_high.to_numpy(dtype=bool)
        if np.any(mask & high) and np.any(mask & ~high):
            supported.append(key)
    return supported


def _raw_excess_gap(frame: pd.DataFrame) -> float | None:
    known = frame.loc[frame.signal_calibrated]
    high = known.loc[known.dispersion_high, "excess_log_loss"].to_numpy(dtype=float)
    low = known.loc[~known.dispersion_high, "excess_log_loss"].to_numpy(dtype=float)
    return float(high.mean() - low.mean()) if len(high) and len(low) else None


def _band_label(index: int) -> str:
    lower, upper = FAVORITE_PROBABILITY_BANDS[index:index + 2]
    closing = "]" if index == len(FAVORITE_PROBABILITY_BANDS) - 2 else ")"
    return f"[{lower:.1f},{upper:.1f}{closing}"


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _profile_records(
    candidate_by_variant: dict[tuple[str, int], pd.DataFrame],
    baseline_by_year: dict[int, pd.DataFrame],
) -> list[dict[str, Any]]:
    records = []
    for variant in VARIANTS:
        for year in YEARS:
            candidate = candidate_by_variant[(variant, year)].copy()
            base = baseline_by_year[year].copy()
            base = base.set_index("golgg_match_id").loc[candidate.golgg_match_id].reset_index()
            for model, frame, probability_column in (
                ("candidate", candidate, "p"), ("canonical_A0", base, "p_full")
            ):
                for dimension, requested_column in PROFILE_COLUMNS.items():
                    source = requested_column
                    if dimension == "experience" and source not in frame and "min_player_series" in frame:
                        source = "min_player_series"
                    group_frame = frame.copy()
                    if dimension == "year":
                        group_frame["_profile_value"] = int(year)
                        source = "model_year"
                    elif source in group_frame:
                        group_frame["_profile_value"] = group_frame[source]
                    else:
                        records.append({
                            "variant": variant, "model_year": year, "model": model,
                            "dimension": dimension, "source_column": None,
                            "value": None, "value_is_missing": True, "sample_size": 0,
                            "status": "diagnostic_missing", "log_loss": None,
                            "brier_score": None, "accuracy": None,
                            "calibration_gap_observed_minus_predicted": None,
                            "tail_ll_ge_2_5": None,
                        })
                        continue
                    for value, group in group_frame.groupby("_profile_value", dropna=False, sort=True):
                        missing_value = bool(pd.isna(value))
                        metric, _ = _score_metrics(group, group[probability_column].to_numpy(dtype=float))
                        records.append({
                            "variant": variant, "model_year": year, "model": model,
                            "dimension": dimension, "source_column": source,
                            "value": None if missing_value else _json_scalar(value),
                            "value_is_missing": missing_value,
                            **metric,
                        })
    return records


def _tournament_block_bootstrap(
    frame: pd.DataFrame, delta: np.ndarray, draws: int
) -> dict[str, Any]:
    work = frame[["model_year", "tournament_name"]].copy()
    labels = [
        (int(year), None if pd.isna(name) else str(name))
        for year, name in work.itertuples(index=False, name=None)
    ]
    block_keys = sorted(set(labels), key=lambda key: (key[0], "" if key[1] is None else key[1]))
    block_index = {key: index for index, key in enumerate(block_keys)}
    row_block = np.fromiter((block_index[key] for key in labels), dtype=int, count=len(labels))
    sums = np.bincount(row_block, weights=np.asarray(delta, dtype=float), minlength=len(block_keys))
    counts = np.bincount(row_block, minlength=len(block_keys)).astype(float)
    observed = float(np.mean(delta))
    result = {
        "delta": observed,
        "equal_block_delta": float(np.mean(sums / counts)),
        "blocks": int(len(block_keys)),
        "rows": int(len(delta)),
        "ci95": None,
        "bootstrap_draws": int(draws),
        "block_unit": "observed tournament_name label × model_year; not certified tournament editions",
    }
    if len(block_keys) < 2:
        return result
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(draws, dtype=float)
    for index in range(draws):
        picks = rng.integers(0, len(block_keys), size=len(block_keys))
        samples[index] = float(sums[picks].sum() / counts[picks].sum())
    result["ci95"] = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
    return result


def _tournament_sensitivity(
    candidate_by_variant: dict[tuple[str, int], pd.DataFrame],
    baseline_by_year: dict[int, pd.DataFrame],
    draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not all("tournament_name" in candidate_by_variant[(variant, year)]
               for variant in VARIANTS for year in YEARS):
        return {"status": "diagnostic_missing", "source_column": "tournament_name"}, []
    summary: dict[str, Any] = {"status": "estimated", "source_column": "tournament_name"}
    block_rows = []
    pooled_by_variant = {}
    for variant in VARIANTS:
        pieces = []
        for year in YEARS:
            test = candidate_by_variant[(variant, year)].copy()
            base = baseline_by_year[year].set_index("golgg_match_id").loc[test.golgg_match_id]
            test["p_full"] = base.p_full.to_numpy(dtype=float)
            test["delta_log_loss"] = (
                losses(test.y.to_numpy(dtype=int), test.p.to_numpy(dtype=float))
                - losses(test.y.to_numpy(dtype=int), test.p_full.to_numpy(dtype=float))
            )
            test["delta_brier"] = (
                np.square(test.p.to_numpy(dtype=float) - test.y.to_numpy(dtype=float))
                - np.square(test.p_full.to_numpy(dtype=float) - test.y.to_numpy(dtype=float))
            )
            pieces.append(test)
            for label, part in test.groupby("tournament_name", dropna=False, sort=True):
                block_rows.append({
                    "comparison": "candidate_vs_canonical_A0",
                    "variant": variant, "model_year": year,
                    "tournament_name": _json_scalar(label),
                    "tournament_label_is_missing": bool(pd.isna(label)),
                    "sample_size": int(len(part)),
                    "delta_log_loss_candidate_minus_canonical_A0": float(part.delta_log_loss.mean()),
                    "delta_brier_candidate_minus_canonical_A0": float(part.delta_brier.mean()),
                    "block_unit": "observed tournament_name label × model_year; not certified tournament editions",
                })
        pooled = pd.concat(pieces, ignore_index=True)
        pooled_by_variant[variant] = pooled
        summary[variant] = {
            "log_loss": _tournament_block_bootstrap(pooled, pooled.delta_log_loss.to_numpy(), draws),
            "brier": _tournament_block_bootstrap(pooled, pooled.delta_brier.to_numpy(), draws),
            "block_unit": "observed tournament_name label × model_year; not certified tournament editions",
            "missing_tournament_label_rows": int(pooled.tournament_name.isna().sum()),
        }

    seed = pooled_by_variant["seed_only"].set_index("golgg_match_id")
    bootstrap = pooled_by_variant["block_bootstrap"].set_index("golgg_match_id")
    if set(seed.index) != set(bootstrap.index):
        raise ValueError("tournament sensitivity requires exactly paired variant match IDs")
    bootstrap = bootstrap.loc[seed.index].copy()
    if not np.array_equal(seed.y.to_numpy(), bootstrap.y.to_numpy()):
        raise ValueError("tournament sensitivity variant labels differ")
    bootstrap["delta_log_loss"] = (
        losses(bootstrap.y.to_numpy(dtype=int), bootstrap.p.to_numpy(dtype=float))
        - losses(seed.y.to_numpy(dtype=int), seed.p.to_numpy(dtype=float))
    )
    bootstrap["delta_brier"] = (
        np.square(bootstrap.p.to_numpy(dtype=float) - bootstrap.y.to_numpy(dtype=float))
        - np.square(seed.p.to_numpy(dtype=float) - seed.y.to_numpy(dtype=float))
    )
    summary["seed_only_vs_block_bootstrap"] = {
        "log_loss": _tournament_block_bootstrap(bootstrap, bootstrap.delta_log_loss.to_numpy(), draws),
        "brier": _tournament_block_bootstrap(bootstrap, bootstrap.delta_brier.to_numpy(), draws),
        "block_unit": "observed tournament_name label × model_year; not certified tournament editions",
    }
    for (year, label), part in bootstrap.groupby(
        ["model_year", "tournament_name"], dropna=False, sort=True
    ):
        block_rows.append({
            "comparison": "block_bootstrap_minus_seed_only",
            "variant": "block_bootstrap", "reference": "seed_only",
            "model_year": int(year), "tournament_name": _json_scalar(label),
            "tournament_label_is_missing": bool(pd.isna(label)),
            "sample_size": int(len(part)),
            "delta_log_loss_candidate_minus_canonical_A0": None,
            "delta_brier_candidate_minus_canonical_A0": None,
            "delta_log_loss_block_bootstrap_minus_seed_only": float(part.delta_log_loss.mean()),
            "delta_brier_block_bootstrap_minus_seed_only": float(part.delta_brier.mean()),
            "block_unit": "observed tournament_name label × model_year; not certified tournament editions",
        })
    return summary, block_rows


def _uncertainty_output_coverage(
    candidate: pd.DataFrame, supplied_columns: set[str]
) -> list[dict[str, Any]]:
    computed = {
        "p_std": "population standard deviation of five members",
        "entropy": "binary entropy of arithmetic mean p",
    }
    optional_diagnostics = (
        "sigma_z", "epistemic_variance", "aleatoric_variance",
        "total_variance", "mutual_information", "p_q10", "p_q90",
    )
    records = []
    for (variant, year, partition), group in candidate.groupby(
        ["variant", "model_year", "partition"], sort=False
    ):
        for diagnostic in (*computed, *optional_diagnostics):
            if diagnostic in computed:
                available = len(group)
                status = "computed_from_members" if diagnostic == "p_std" else "computed_from_mean_p"
                source = computed[diagnostic]
            elif diagnostic not in supplied_columns or diagnostic not in group:
                available = 0
                status = "diagnostic_missing"
                source = diagnostic
            else:
                available = int(group[diagnostic].notna().sum())
                status = "available" if available == len(group) else (
                    "partially_missing" if available else "diagnostic_missing"
                )
                source = diagnostic
            records.append({
                "variant": variant, "model_year": int(year), "partition": partition,
                "diagnostic": diagnostic, "source_column": source,
                "available_rows": int(available),
                "missing_rows": int(len(group) - available),
                "status": status,
            })
    return records


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def evaluate_uncertainty(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    output_dir: Path,
    *,
    draws: int = 5000,
) -> dict[str, Any]:
    """Evaluate both annual arms against the complete canonical A0 TEST cohort.

    ``predictions`` contains the six annual arm tables concatenated together;
    ``baseline`` is the locked canonical joined cohort with ``p_full``. CAL rows
    set all selection/signal cutoffs. TEST outcomes never alter those cutoffs.
    """
    if draws < BOOTSTRAP_DRAWS_MIN:
        raise ValueError(f"draws must be at least {BOOTSTRAP_DRAWS_MIN} by the frozen protocol")
    canonical, candidate, candidate_by_variant = _check_candidate_coverage_and_identity(predictions, baseline)
    cal_by_variant = {
        (variant, year): candidate.loc[
            candidate.variant.eq(variant)
            & candidate.model_year.eq(year)
            & candidate.partition.eq("cal")
        ].copy()
        for variant in VARIANTS for year in YEARS
    }

    metrics: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    selection_coverage: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    candidate_test_frames: dict[tuple[str, int], pd.DataFrame] = {}
    baseline_by_year = {year: canonical.loc[canonical.model_year.eq(year)].copy() for year in YEARS}

    for variant in VARIANTS:
        for year in YEARS:
            cal = cal_by_variant[(variant, year)]
            test = candidate_by_variant[(variant, year)].copy()
            candidate_test_frames[(variant, year)] = test
            aligned_baseline = baseline_by_year[year].set_index("golgg_match_id").loc[test.golgg_match_id].reset_index()
            selected_masks: dict[tuple[str, float], np.ndarray] = {}

            for selector, score_column in (("p_std", "p_std"), ("entropy", "entropy")):
                for retention in RETENTION_LEVELS:
                    cutoff = _fit_cutoff(cal, score_column, retention)
                    selected, test_detail = _select_from_cutoff(test, score_column, cutoff)
                    selected_masks[(selector, retention)] = selected
                    thresholds.append({
                        "variant": variant, "model_year": year, "selector": selector,
                        "retention": float(retention), **cutoff,
                    })
                    selection_coverage.append({
                        "variant": variant, "model_year": year, "partition": "test",
                        "selector": selector, "retention": float(retention),
                        "cal_threshold": cutoff["threshold"], **test_detail,
                    })
                    for row in test.loc[selected, ["golgg_match_id"]].itertuples(index=False):
                        selections.append({
                            "variant": variant, "model_year": year, "partition": "test",
                            "selector": selector, "retention": float(retention),
                            "golgg_match_id": row.golgg_match_id,
                            "threshold": cutoff["threshold"],
                        })

            for selector, retention in selected_masks:
                selected = selected_masks[(selector, retention)]
                metadata = {
                    "variant": variant, "model_year": year, "partition": "test",
                    "selector": selector, "retention": float(retention),
                    "selected_rows": int(selected.sum()),
                    "selected_coverage": float(selected.mean()),
                }
                selected_frame = test.loc[selected].reset_index(drop=True)
                selected_baseline = aligned_baseline.loc[selected].reset_index(drop=True)
                _add_score_records(
                    selected_frame, selected_baseline, "p",
                    metadata={**metadata, "mode": "full"}, draws=draws,
                    metrics=metrics, comparisons=comparisons, calibration=calibration,
                )
                for mode, probability_column in OPTIONAL_MODES.items():
                    if probability_column not in test:
                        if not any(
                            item["variant"] == variant and item["model_year"] == year
                            and item["partition"] == "test" and item["diagnostic"] == mode
                            for item in diagnostics
                        ):
                            diagnostics.append({
                                "variant": variant, "model_year": year, "partition": "test",
                                "diagnostic": mode, "source_column": probability_column,
                                "available_rows": 0, "missing_rows": int(len(test)),
                                "status": "diagnostic_missing",
                            })
                        continue
                    available = selected & test[probability_column].notna().to_numpy()
                    mode_frame = test.loc[available].reset_index(drop=True)
                    mode_baseline = aligned_baseline.loc[available].reset_index(drop=True)
                    if not any(
                        item["variant"] == variant and item["model_year"] == year
                        and item["partition"] == "test" and item["diagnostic"] == mode
                        for item in diagnostics
                    ):
                        diagnostics.append({
                            "variant": variant, "model_year": year, "partition": "test",
                            "diagnostic": mode, "source_column": probability_column,
                            "available_rows": int(test[probability_column].notna().sum()),
                            "missing_rows": int(test[probability_column].isna().sum()),
                            "status": "available" if test[probability_column].notna().all() else "partially_missing",
                        })
                    _add_score_records(
                        mode_frame, mode_baseline, probability_column,
                        metadata={**metadata, "mode": mode,
                                      "mode_available_rows": int(available.sum())},
                        draws=draws, metrics=metrics, comparisons=comparisons,
                        calibration=calibration,
                    )

    variant_comparison_rows, variant_comparison = _paired_variant_comparisons(candidate_test_frames, draws)
    comparisons.extend(variant_comparison_rows)
    signal_report, signal_band_rows = _signal_tables_and_report(candidate_test_frames, cal_by_variant, draws)
    tournament_report, tournament_rows = _tournament_sensitivity(
        candidate_test_frames, baseline_by_year, draws
    )
    profiles = _profile_records(candidate_test_frames, baseline_by_year)
    ll_delta = variant_comparison["log_loss"]
    ll_upper = ll_delta["ci95"][1] if ll_delta["ci95"] is not None else None
    primary_signal = signal_report["block_bootstrap"]["overall"]
    signal_lower = (
        primary_signal["ci95"][0]
        if primary_signal["status"] == "estimated" and primary_signal["ci95"] is not None
        else None
    )
    if ll_upper is None or signal_lower is None:
        verdict = "NOT_EVALUABLE"
    elif ll_upper <= LL_NONINFERIORITY_MARGIN and signal_lower > 0.0:
        verdict = "RESEARCH_CRITERION_MET_DEVELOPMENT_ONLY"
    else:
        verdict = "RESEARCH_CRITERION_NOT_MET"

    mode_coverage = []
    for (variant, year), test in candidate_test_frames.items():
        for mode, column in OPTIONAL_MODES.items():
            present = test[column].notna() if column in test else pd.Series(False, index=test.index)
            mode_coverage.append({
                "variant": variant, "model_year": year, "partition": "test",
                "mode": mode, "source_column": column,
                "available_rows": int(present.sum()), "missing_rows": int((~present).sum()),
                "status": "available" if present.all() else ("partially_missing" if present.any() else "diagnostic_missing"),
            })
        cal = cal_by_variant[(variant, year)]
        for mode, column in OPTIONAL_MODES.items():
            present = cal[column].notna() if column in cal else pd.Series(False, index=cal.index)
            mode_coverage.append({
                "variant": variant, "model_year": year, "partition": "cal",
                "mode": mode, "source_column": column,
                "available_rows": int(present.sum()), "missing_rows": int((~present).sum()),
                "status": "available" if present.all() else ("partially_missing" if present.any() else "diagnostic_missing"),
            })

    diagnostic_coverage = _uncertainty_output_coverage(candidate, set(predictions.columns))
    diagnostics.extend(diagnostic_coverage)
    profile_diagnostics = {}
    for dimension, requested_column in PROFILE_COLUMNS.items():
        sources = set()
        for variant in VARIANTS:
            for year in YEARS:
                frame = candidate_test_frames[(variant, year)]
                source = requested_column
                if dimension == "experience" and source not in frame and "min_player_series" in frame:
                    source = "min_player_series"
                if source in frame:
                    sources.add(source)
        profile_diagnostics[dimension] = {
            "source_columns": sorted(sources),
            "available": bool(sources),
            "missing_is_preserved": True,
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "DEVELOPMENT_ONLY",
        "method": "frozen Causal A0 uncertainty protocol; canonical benchmark remains headline SSOT",
        "promotion": "NOT_EVALUATED",
        "production_promotion": False,
        "canonical_rows": int(len(canonical)),
        "canonical_test_rows_by_year": {str(year): int(len(baseline_by_year[year])) for year in YEARS},
        "arms": list(VARIANTS),
        "model_years": list(YEARS),
        "draws": int(draws),
        "retention_levels": list(RETENTION_LEVELS),
        "favorite_probability_bands": [_band_label(i) for i in range(len(FAVORITE_PROBABILITY_BANDS) - 1)],
        "selection_policy": {
            "threshold_source": "each variant/model-year CAL distribution only",
            "p_std": "retain lower population member spread; epistemic selector only, not total confidence",
            "entropy": "retain lower predictive binary entropy; entropy and spread are reported as distinct selectors",
            "calibration_ties": "ascending golgg_match_id fixes CAL selection at the requested count",
            "test_threshold_ties": "apply the CAL tied-selection fraction; ascending golgg_match_id resolves TEST ties",
            "equal_members_at_p_0_5": "zero spread remains distinct from maximum predictive entropy; it is not labeled as calibrated confidence",
        },
        "primary_research_criterion": {
            "paired_log_loss_delta": "block_bootstrap minus seed_only, paired on all 11,550 canonical TEST rows",
            "log_loss_noninferiority_margin": LL_NONINFERIORITY_MARGIN,
            "required_log_loss_upper_95": f"<= {LL_NONINFERIORITY_MARGIN}",
            "required_adjusted_excess_loss_lower_95": "> 0 for block_bootstrap",
            "criterion_verdict": verdict,
            "verdict_interpretation": (
                "Both preregistered research conditions are met on development data only."
                if verdict == "RESEARCH_CRITERION_MET_DEVELOPMENT_ONLY"
                else "No demonstrated added value under the preregistered criterion; this is not proof that ensemble dispersion is intrinsically invalid."
            ),
            "log_loss_upper_95": ll_upper,
            "adjusted_excess_loss_lower_95": signal_lower,
            "development_only_no_production_promotion": True,
        },
        "paired_variant_scores": variant_comparison,
        "thresholds": thresholds,
        "selection_coverage": selection_coverage,
        "adjusted_excess_loss": signal_report,
        "tournament_label_year_sensitivity": tournament_report,
        "diagnostics": {
            "p_std_source": "population_std_of_five_members",
            "entropy_source": "binary predictive entropy of arithmetic mean p, nats",
            "optional_mode_coverage": mode_coverage,
            "profile_diagnostics": profile_diagnostics,
            "uncertainty_output_coverage": diagnostic_coverage,
        },
        "limitations": [
            "Development data repeatedly examined across 2024-2026; not prospective confirmation.",
            "Member spread is conditional on fixed historical features and shared calibration; common biases, shared feature-state uncertainty, and calibration uncertainty are absent.",
            "Member quantiles are descriptive only and are not confidence intervals.",
            "Tournament names are observed labels, not certified tournament editions.",
            "No market inputs, application/EV analysis, live serving, or production promotion is evaluated.",
        ],
    }

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=False, exist_ok=False)
    tables = {
        "metrics.csv": metrics,
        "comparisons.csv": comparisons,
        "calibration.csv": calibration,
        "thresholds.csv": thresholds,
        "selection_coverage.csv": selection_coverage,
        "selections.csv": selections,
        "favorite_band_signal.csv": signal_band_rows,
        "profiles.csv": profiles,
        "diagnostics.csv": diagnostics,
        "uncertainty_diagnostics.csv": diagnostic_coverage,
        "tournament_sensitivity.csv": tournament_rows,
    }
    report["artifacts"] = {
        "output_dir": str(output),
        **{name: str(output / name) for name in tables},
        "summary": str(output / "summary.json"),
    }
    for filename, rows in tables.items():
        pd.DataFrame(_json_safe(rows)).to_csv(output / filename, index=False)
    (output / "summary.json").write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _json_safe(report)
