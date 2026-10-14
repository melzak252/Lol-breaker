
"""Pure context extraction and persistence/evaluation diagnostics for C0."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analysis.metrics import binary_log_loss_vector
from src.analysis.production_market_benchmark import paired_cluster_interval

CONTEXTS = (
    "low_statistics",
    "novel_roster_support",
    "high_sports_disagreement",
)
CONTEXT_COLUMNS = tuple(f"x_{name}" for name in CONTEXTS)
CONFIDENCE_EDGES = np.asarray([.5, .6, .7, .8, .9, 1.0], dtype=float)
DETECTION_MONTHS = tuple(pd.period_range("2024-07", "2024-12", freq="M").astype(str))
REPLICATION_MONTHS = tuple(pd.period_range("2025-01", "2025-12", freq="M").astype(str))
PERSISTENCE_CONFIDENCE = 0.9916666666666667
EVALUATION_CONFIDENCE = 0.9833333333333333
DRAWS = 5000
SEED = 20261004
PRIMARY_MONTHS = tuple(pd.period_range("2026-01", "2026-09", freq="M").astype(str))
DISAGREEMENT_TIE_ATOL = 1e-12
NEUTRAL_TOLERANCE = 1e-6

def _monthly_interval(
    deltas: np.ndarray,
    months: np.ndarray,
    *,
    draws: int,
    seed: int,
    confidence_level: float,
    month_universe: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=float)
    labels = np.asarray(months, dtype=object).astype(str)
    if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("monthly interval values and dates must align")
    if not np.isfinite(values).all():
        raise ValueError("monthly interval differences must be finite")
    universe = tuple(month_universe or sorted(set(labels)))
    if not universe or set(labels).difference(universe):
        raise ValueError("monthly interval labels must lie in a nonempty fixed universe")
    sums_by_month = {month: 0.0 for month in universe}
    counts_by_month = {month: 0 for month in universe}
    for value, month in zip(values, labels):
        sums_by_month[month] += float(value)
        counts_by_month[month] += 1
    sums = np.asarray([sums_by_month[month] for month in universe], dtype=float)
    counts = np.asarray([counts_by_month[month] for month in universe], dtype=float)
    active = counts > 0
    estimate = float(sums.sum() / counts.sum()) if counts.sum() else None
    equal_estimate = float(np.mean(sums[active] / counts[active])) if active.any() else None
    lower = upper = equal_lower = equal_upper = None
    valid = 0
    undefined = 0
    if len(universe) >= 2 and estimate is not None:
        rng = np.random.default_rng(seed)
        selected = rng.integers(0, len(universe), size=(draws, len(universe)))
        denominators = counts[selected].sum(axis=1)
        defined = denominators > 0
        valid = int(defined.sum())
        undefined = int(draws - valid)
        if undefined == 0:
            samples = sums[selected].sum(axis=1) / denominators
            selected_active = counts[selected] > 0
            per_month = np.divide(sums[selected], counts[selected],
                                  out=np.zeros_like(sums[selected]), where=selected_active)
            equal_samples = np.sum(per_month, axis=1) / selected_active.sum(axis=1)
            tail = (1 - confidence_level) / 2
            lower, upper = (float(value) for value in np.quantile(samples, [tail, 1-tail]))
            equal_lower, equal_upper = (float(value) for value in np.quantile(equal_samples, [tail, 1-tail]))
    return {
        "n": int(len(values)), "blocks": len(universe), "delta": estimate,
        "ci_lower": lower, "ci_upper": upper, "equal_block_delta": equal_estimate,
        "equal_block_ci_lower": equal_lower, "equal_block_ci_upper": equal_upper,
        "draws": int(draws), "seed": int(seed), "confidence_level": float(confidence_level),
        "active_blocks": int(active.sum()), "valid_replicates": valid,
        "undefined_replicates": undefined,
        "status": "ESTIMATED" if lower is not None else "INSUFFICIENT_BLOCKS_OR_UNDEFINED",
    }


def _safe(value: Any) -> Any:
    """Recursively convert numpy/pandas values to strict finite JSON values."""
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return value


def _required(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    absent = sorted(columns.difference(frame.columns))
    if absent:
        raise ValueError(f"{name} is missing required columns: {absent}")


def _dates(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{name} contains invalid or missing dates")
    return result


def _probabilities(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values <= 0) | (values >= 1)).any():
            raise ValueError(f"{column} must contain finite probabilities strictly inside (0, 1)")


def extract_context_features(
    frame: pd.DataFrame, disagreement_threshold: float | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add three ternary support flags; missing covariates remain UNKNOWN/NaN.

    When no threshold is supplied, the high-sports-disagreement cutoff is the
    median of finite 2024 expert-probability spreads. Labels are never consulted.
    """
    required = {
        "golgg_match_id", "date", "label_release_day", "team1_id", "team2_id",
        "best_of", "y", "p_c0", "p_a0", "p_neural", "p_rating", "p_mlp",
        "history_stat_coverage", "unknown_players", "history_token_fraction",
        "tournament_name", "competition_tier",
    }
    _required(frame, required, "context feature frame")
    if frame.empty or frame.golgg_match_id.isna().any() or frame.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("context feature frame must have unique nonmissing match identities")
    result = frame.copy()
    result["date"] = _dates(result.date, "date")
    result["label_release_day"] = _dates(result.label_release_day, "label_release_day")
    for column in ("p_c0", "p_a0", "p_neural", "p_rating", "p_mlp"):
        raw = result[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        values = numeric.to_numpy(float)
        if (raw.notna() & numeric.isna()).any() or np.isinf(values).any():
            raise ValueError(f"{column} contains a malformed or infinite probability")
        result[column] = numeric
        if column in ("p_c0", "p_a0") and not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite for every row")
        finite_values = values[np.isfinite(values)]
        if ((finite_values <= 0) | (finite_values >= 1)).any():
            raise ValueError(f"{column} must contain probabilities strictly inside (0, 1) when present")
    y = pd.to_numeric(result.y, errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("outcomes must be binary and present")
    result["y"] = y.astype(int)

    coverage = pd.to_numeric(result.history_stat_coverage, errors="coerce")
    if (coverage.notna() & ~coverage.between(0, 1, inclusive="both")).any():
        raise ValueError("history_stat_coverage must be within [0, 1] when present")
    unknown = pd.to_numeric(result.unknown_players, errors="coerce")
    if (unknown.notna() & (unknown < 0)).any():
        raise ValueError("unknown_players cannot be negative")
    token = pd.to_numeric(result.history_token_fraction, errors="coerce")
    if (token.notna() & ~token.between(0, 1, inclusive="both")).any():
        raise ValueError("history_token_fraction must be within [0, 1] when present")

    result["x_low_statistics"] = np.where(coverage.isna(), np.nan, coverage.lt(.90).astype(float))
    result["x_novel_roster_support"] = np.where(unknown.isna(), np.nan, unknown.gt(0).astype(float))
    experts = result[["p_neural", "p_rating", "p_mlp"]]
    spreads = experts.max(axis=1, skipna=False) - experts.min(axis=1, skipna=False)
    if disagreement_threshold is None:
        cutoff_mask = result.date.dt.year.eq(2024) & np.isfinite(spreads.to_numpy(float))
        cutoff_values = spreads.loc[cutoff_mask]
        if cutoff_values.empty:
            raise ValueError("cannot derive sports-disagreement threshold without finite 2024 covariates")
        threshold = float(cutoff_values.median())
        threshold_source = "median_finite_2024_covariate_spreads"
    else:
        threshold = float(disagreement_threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("disagreement_threshold must be finite and nonnegative")
        threshold_source = "explicit_frozen_threshold"
    result["x_high_sports_disagreement"] = np.where(
        np.isfinite(spreads.to_numpy(float)),
        spreads.gt(threshold + DISAGREEMENT_TIE_ATOL).astype(float), np.nan,
    )
    metadata = {
        "high_sports_disagreement_threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_population": "all finite 2024 covariate-only neural/rating/MLP spreads",
        "disagreement_tie_atol": DISAGREEMENT_TIE_ATOL,
        "disagreement_rule": "risk only when expert spread > cutoff + tie tolerance; ties are reference",
        "context_names": list(CONTEXTS),
        "feature_columns": list(CONTEXT_COLUMNS),
        "unknown_policy": "missing support covariate remains NaN/UNKNOWN; no missingness effect is invented",
        "novel_roster_support_interpretation": "unknown-player novelty/support only; roster turnover is unassessed",
        "history_token_fraction_role": "mask-derived descriptive support only; not a tested context dimension",
        "missingness": {
            column: int(result[column].isna().sum())
            for column in ("history_stat_coverage", "unknown_players", "history_token_fraction")
        },
    }
    return result, _safe(metadata)


def _bin_index(favorite: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(CONFIDENCE_EDGES, favorite, side="right") - 1, 0, 4)


def _ratio_bootstrap(
    rows: pd.DataFrame,
    context_column: str,
    bins: list[int],
    weights: dict[int, float],
    months: tuple[str, ...],
    *,
    draws: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Bootstrap the fixed-bin weighted risk-minus-reference slope contrast."""
    nmonths, nbins = len(months), len(bins)
    month_index = {month: i for i, month in enumerate(months)}
    sums = {group: np.zeros((nmonths, nbins), dtype=float) for group in (0, 1)}
    counts = {group: np.zeros((nmonths, nbins), dtype=float) for group in (0, 1)}
    month_values = rows["month"].astype(str).to_numpy()
    state_values = pd.to_numeric(rows[context_column], errors="coerce").to_numpy(float)
    bin_values = rows["_confidence_bin"].to_numpy(dtype=int)
    residual_values = rows["_u"].to_numpy(dtype=float)
    for month, group_value, bin_value, residual in zip(
        month_values, state_values, bin_values, residual_values
    ):
        if month not in month_index:
            continue
        if not np.isfinite(group_value) or group_value not in (0.0, 1.0):
            continue
        if bin_value not in bins:
            continue
        bin_idx = bins.index(bin_value)
        m_idx = month_index[month]
        sums[int(group_value)][m_idx, bin_idx] += residual
        counts[int(group_value)][m_idx, bin_idx] += 1.0

    def statistic(sampled_months: np.ndarray) -> float | None:
        risk_counts = counts[1][sampled_months].sum(axis=0)
        reference_counts = counts[0][sampled_months].sum(axis=0)
        if np.any(risk_counts <= 0) or np.any(reference_counts <= 0):
            return None
        risk_means = sums[1][sampled_months].sum(axis=0) / risk_counts
        reference_means = sums[0][sampled_months].sum(axis=0) / reference_counts
        return float(sum(weights[bin_value] * (risk_means[index] - reference_means[index])
                         for index, bin_value in enumerate(bins)))

    observed = statistic(np.arange(nmonths, dtype=int))
    if observed is None:
        return {"estimate": None, "ci_lower": None, "ci_upper": None, "draws": draws,
                "seed": seed, "months": nmonths, "status": "ZERO_DENOMINATOR"}
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    undefined_replicates = 0
    for _ in range(draws):
        value = statistic(rng.integers(0, nmonths, size=nmonths))
        if value is None:
            undefined_replicates += 1
        else:
            samples.append(value)
    if undefined_replicates:
        return {"estimate": observed, "ci_lower": None, "ci_upper": None,
                "draws": draws, "seed": seed, "months": nmonths,
                "valid_replicates": draws - undefined_replicates,
                "undefined_replicates": undefined_replicates,
                "status": "UNDEFINED_REPLICATES"}
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(np.asarray(samples), [tail, 1.0 - tail])
    return {"estimate": observed, "ci_lower": float(lower), "ci_upper": float(upper),
            "draws": draws, "seed": seed, "months": nmonths,
            "valid_replicates": draws, "undefined_replicates": 0,
            "confidence_level": confidence_level, "status": "ESTIMATED"}


def _period_support(
    rows: pd.DataFrame, context_column: str, minimum_rows: int,
    minimum_months: int, minimum_per_bin: int,
) -> tuple[dict[str, Any], list[int], dict[int, float]]:
    contexts = pd.to_numeric(rows[context_column], errors="coerce")
    supported_bins: list[int] = []
    raw_weights: dict[int, float] = {}
    bin_report: list[dict[str, Any]] = []
    for bin_value in range(5):
        in_bin = rows["_confidence_bin"].eq(bin_value)
        risk_count = int((in_bin & contexts.eq(1)).sum())
        reference_count = int((in_bin & contexts.eq(0)).sum())
        support = risk_count >= minimum_per_bin and reference_count >= minimum_per_bin
        bin_report.append({"bin": bin_value, "low": float(CONFIDENCE_EDGES[bin_value]),
                           "high": float(CONFIDENCE_EDGES[bin_value + 1]),
                           "risk_rows": risk_count, "reference_rows": reference_count,
                           "common_support": bool(support)})
        if support:
            supported_bins.append(bin_value)
            raw_weights[bin_value] = float(risk_count + reference_count)
    total = sum(raw_weights.values())
    weights = {key: value / total for key, value in raw_weights.items()} if total else {}
    risk = contexts.eq(1)
    reference = contexts.eq(0)
    risk_months = int(rows.loc[risk, "month"].nunique())
    reference_months = int(rows.loc[reference, "month"].nunique())
    risk_rows, reference_rows = int(risk.sum()), int(reference.sum())
    pass_support = (risk_rows >= minimum_rows and reference_rows >= minimum_rows
                    and risk_months >= minimum_months and reference_months >= minimum_months
                    and bool(supported_bins))
    report = {
        "risk_rows": risk_rows, "reference_rows": reference_rows,
        "risk_months": risk_months, "reference_months": reference_months,
        "minimum_group_rows_per_year": minimum_rows,
        "minimum_group_months_per_year": minimum_months,
        "minimum_rows_per_bin_and_group": minimum_per_bin,
        "bins": bin_report, "supported_bins": supported_bins,
        "bin_weights": {str(key): value for key, value in weights.items()},
        "support_pass": bool(pass_support),
    }
    return report, supported_bins, weights


def _period_rows(
    prepared: pd.DataFrame, start: str, end: str,
) -> pd.DataFrame:
    dates = prepared.date
    keep = dates.ge(pd.Timestamp(start)) & dates.lt(pd.Timestamp(end))
    # Neutral C0 rows are excluded only from residual/slope diagnostics.
    keep &= (prepared.p_c0 - .5).abs().gt(NEUTRAL_TOLERANCE)
    keep &= prepared["_global_match"]
    keep &= prepared["_valid_global"]
    return prepared.loc[keep].copy()

def _history_missingness(frame: pd.DataFrame) -> dict[str, Any]:
    report = {}
    for name, start, end in (
        ("detection", "2024-07-01", "2025-01-01"),
        ("replication", "2025-01-01", "2026-01-01"),
    ):
        part = frame.loc[frame.date.ge(start) & frame.date.lt(end)]
        report[name] = {
            "rows": len(part),
            "global_forecast_missing": int((~part["_global_match"]).sum()),
            "neutral_c0_rows": int((part.p_c0 - .5).abs().le(NEUTRAL_TOLERANCE).sum()),
            "unknown_context_rows": {
                context: int(pd.to_numeric(part[column], errors="coerce").isna().sum())
                for context, column in zip(CONTEXTS, CONTEXT_COLUMNS)
            },
        }
    return report

def _persistence_confounding_report(frame: pd.DataFrame) -> dict[str, Any]:
    dimension_sources = [
        (column, column) for column in (
            "best_of", "competition_tier", "recipe_year", "calibration_year",
        ) if column in frame
    ]
    if "model_year" in frame:
        dimension_sources.append(("native_model_year_recipe_year", "model_year"))
    report: dict[str, Any] = {
        "proxy_caveat": "Tournament-name/year is a support proxy, not a certified edition or causal adjustment.",
        "periods": {},
        "causal_claim": False,
        "turnover_claim": False,
        "dimension_interpretation": {
            "native_model_year_recipe_year": (
                "Native artifact model_year, interpreted only as its recipe year; "
                "not CAL label year or an isolated epoch effect."
            ),
            "calibration_year": (
                "UNASSESSED: calibration_year is absent."
                if "calibration_year" not in frame else
                "Observed source calibration_year field; no causal interpretation."
            ),
        },
    }
    for label, start, end in (
        ("detection_2024h2", "2024-07-01", "2025-01-01"),
        ("replication_2025", "2025-01-01", "2026-01-01"),
    ):
        part = frame.loc[frame.date.ge(start) & frame.date.lt(end)].copy()
        by_dimension = {}
        for report_name, column in dimension_sources:
            by_dimension[report_name] = [
                {"level": _safe(key), "rows": int(len(group))}
                for key, group in part.groupby(column, dropna=False, sort=True)
            ]
        if "unknown_players" in part:
            unknown = pd.to_numeric(part.unknown_players, errors="coerce")
            by_dimension["source_unseen_support"] = [
                {"level": level, "rows": int(count)}
                for level, count in (
                    ("UNKNOWN", unknown.isna().sum()),
                    ("unseen", unknown.gt(0).sum()),
                    ("seen", unknown.eq(0).sum()),
                )
            ]
        if "tournament_name" in part:
            name = part.tournament_name.fillna("UNKNOWN").astype(str)
            year = part.date.dt.year.astype(str)
            proxy = name + "|" + year
            by_dimension["source_name_year_proxy"] = [
                {"proxy": str(key), "rows": int(len(group))}
                for key, group in part.groupby(proxy, sort=True)
            ]
        report["periods"][label] = {
            "rows": len(part),
            "by_dimension": by_dimension,
            "unknown_support_rows": {
                column: int(pd.to_numeric(part[column], errors="coerce").isna().sum())
                for column in ("history_stat_coverage", "unknown_players", "history_token_fraction")
                if column in part
            },
        }
    return report


def evaluate_persistence(
    frame: pd.DataFrame,
    global_predictions: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate context-dependent reliability persistence without any new fit."""
    cfg = dict(config or {})
    draws = int(cfg.get("draws", DRAWS))
    seed = int(cfg.get("seed", SEED))
    min_rows = int(cfg.get("minimum_group_rows_per_year", 200))
    min_months = int(cfg.get("minimum_group_months_per_year", 6))
    min_bin = int(cfg.get("minimum_risk_and_reference_rows_per_bin", 20))
    level = float(cfg.get("confidence_level", PERSISTENCE_CONFIDENCE))
    if draws <= 0 or min_rows < 1 or min_months < 1 or min_bin < 1 or not 0 < level < 1:
        raise ValueError("invalid persistence evaluation configuration")
    _required(frame, {"golgg_match_id", "date", "label_release_day", "p_c0", "y", *CONTEXT_COLUMNS}, "persistence frame")
    _required(global_predictions, {"golgg_match_id", "date", "origin", "q_global"}, "prior global predictions")
    if frame.golgg_match_id.isna().any() or frame.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("persistence frame match identities must be unique")
    gp = global_predictions.copy()
    if gp.golgg_match_id.isna().any() or gp.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("global forecasts must have unique nonmissing match identities")
    source = frame.copy()
    source["date"] = _dates(source.date, "date")
    source["label_release_day"] = _dates(source.label_release_day, "label_release_day")
    source["p_c0"] = pd.to_numeric(source.p_c0, errors="coerce")
    source["y"] = pd.to_numeric(source.y, errors="coerce")
    if source.p_c0.isna().any() or not source.p_c0.between(0, 1, inclusive="neither").all():
        raise ValueError("p_c0 must be a finite interior probability")
    if source.y.isna().any() or not source.y.isin([0, 1]).all():
        raise ValueError("persistence outcomes must be binary")
    for column in CONTEXT_COLUMNS:
        values = pd.to_numeric(source[column], errors="coerce")
        if (values.notna() & ~values.isin([0, 1])).any():
            raise ValueError(f"{column} must contain only 0, 1 or UNKNOWN")
        source[column] = values
    source["_key"] = source.golgg_match_id.astype(str)
    gp["_key"] = gp.golgg_match_id.astype(str)
    source["_date_key"] = source.date.astype("datetime64[ns]")
    gp["_date_key"] = _dates(gp.date, "global prediction date").astype("datetime64[ns]")
    gp["origin"] = _dates(gp.origin, "global fit origin")
    gp["q_global"] = pd.to_numeric(gp.q_global, errors="coerce")
    historical_gp = gp.copy()
    if not gp["_date_key"].ge(pd.Timestamp("2024-07-01")).all() or not gp["_date_key"].lt(pd.Timestamp("2026-01-01")).all():
        raise ValueError("global predictions may cover only 2024H2 and 2025")
    if historical_gp.q_global.isna().any() or not historical_gp.q_global.between(0, 1, inclusive="neither").all():
        raise ValueError("historical global controls must be finite interior probabilities")
    if (historical_gp.origin > pd.to_datetime(historical_gp.date)).any():
        raise ValueError("global forecast origin cannot follow the target event date")
    expected = source.loc[
        source.date.ge("2024-07-01") & source.date.lt("2026-01-01"),
        ["_key", "_date_key"],
    ]
    expected_pairs = set(map(tuple, expected.to_numpy()))
    supplied_pairs = set(map(tuple, historical_gp[["_key", "_date_key"]].to_numpy()))
    if supplied_pairs != expected_pairs:
        raise ValueError("prior global predictions must cover exactly the eligible 2024H2/2025 identities and dates")
    if not historical_gp["_date_key"].isin(source["_date_key"]).all():
        raise ValueError("global prediction dates do not match source rows")
    joined = source.merge(
        historical_gp[["_key", "_date_key", "origin", "q_global"]],
        on=["_key", "_date_key"], how="left", validate="one_to_one", indicator="_global_match",
    )
    joined["_global_match"] = joined["_global_match"].eq("both")
    # No 2026 forecast, outcome or context enters the gate or the slope statistics.
    joined["_valid_global"] = joined.q_global.notna() & joined.origin.le(joined.date)
    favorite = np.maximum(joined.p_c0.to_numpy(float), 1.0 - joined.p_c0.to_numpy(float))
    joined["_confidence_bin"] = _bin_index(favorite)
    joined["month"] = joined.date.dt.to_period("M").astype(str)
    z = np.log(joined.p_c0.to_numpy(float) / (1.0 - joined.p_c0.to_numpy(float)))
    joined["_u"] = (joined.q_global.to_numpy(float) - joined.y.to_numpy(float)) * z
    # _u for unmatched rows is NaN; it is never retained by _period_rows.
    periods = {
        "detection": ("2024-07-01", "2025-01-01", DETECTION_MONTHS),
        "replication": ("2025-01-01", "2026-01-01", REPLICATION_MONTHS),
    }
    contexts: dict[str, Any] = {}
    flat_claims: list[dict[str, Any]] = []
    for name, col in zip(CONTEXTS, CONTEXT_COLUMNS):
        period_rows = {
            label: _period_rows(joined, start, end)
            for label, (start, end, _) in periods.items()
        }
        detection_support, bins, weights = _period_support(
            period_rows["detection"], col, min_rows, min_months, min_bin,
        )
        # Replication must support exactly the 2024H2 common bins; no bins are dropped.
        replication_support, _, _ = _period_support(
            period_rows["replication"], col, min_rows, min_months, min_bin,
        )
        replication_required = all(
            next(item for item in replication_support["bins"] if item["bin"] == index)["common_support"]
            for index in bins
        ) if bins else False
        replication_support["required_detection_bins_supported"] = bool(replication_required)
        stats = {}
        for period_name, (_, _, universe) in periods.items():
            part = period_rows[period_name]
            support = detection_support if period_name == "detection" else replication_support
            if not bins or not support["support_pass"] or (period_name == "replication" and not replication_required):
                interval = {"estimate": None, "ci_lower": None, "ci_upper": None,
                            "draws": draws, "seed": seed, "months": len(universe),
                            "confidence_level": level,
                            "status": "INSUFFICIENT_SUPPORT"}
            else:
                interval = _ratio_bootstrap(
                    part, col, bins, weights, universe, draws=draws,
                    seed=seed,
                    confidence_level=level,
                )
            stats[period_name] = {**interval, "support": support,
                                  "month_universe": list(universe)}
            flat_claims.append({"context": name, "period": period_name,
                                "slope_contrast": interval.get("estimate"),
                                "confidence_level": level, **interval})
        detection = stats["detection"]
        replication = stats["replication"]
        estimates = (detection.get("estimate"), replication.get("estimate"))
        sign_match = (all(value is not None and np.isfinite(value) and value != 0 for value in estimates)
                      and np.sign(estimates[0]) == np.sign(estimates[1]))
        significant = all(
            stats[period].get("status") == "ESTIMATED"
            and stats[period].get("ci_lower") is not None
            and (stats[period]["ci_lower"] > 0 or stats[period]["ci_upper"] < 0)
            for period in periods
        )
        context_pass = bool(detection_support["support_pass"] and replication_support["support_pass"]
                            and replication_required and sign_match and significant)
        contexts[name] = {
            "detection": detection, "replication": replication,
            "same_nonzero_sign": bool(sign_match),
            "both_intervals_exclude_zero": bool(significant),
            "gate_pass": context_pass,
        }
    gate_passers = [name for name in CONTEXTS if contexts[name]["gate_pass"]]
    supported_contexts = [
        name for name in CONTEXTS
        if contexts[name]["detection"]["support"]["support_pass"]
        and contexts[name]["replication"]["support"]["support_pass"]
        and contexts[name]["replication"]["support"]["required_detection_bins_supported"]
    ]
    gate_status = (
        "PASS" if gate_passers else
        "PERSISTENCE_NOT_CONFIRMED" if supported_contexts else
        "INSUFFICIENT_SUPPORT"
    )
    # Descriptive favorite calibration gap only; it does not trigger any fit.
    favorite_gap = _favorite_gap(joined, contexts, draws, seed)
    return _safe({
        "status": "COMPLETE", "gate": {
            "pass": bool(gate_passers), "status": gate_status,
            "passing_contexts": gate_passers, "supported_contexts": supported_contexts,
            "rule": "at least one context has support, same nonzero contrast sign, and both 99.1667% intervals exclude zero",
            "uses_2026_outcomes": False,
        },
        "contexts": contexts, "claims": flat_claims,
        "favorite_gap": favorite_gap,
        "configuration": {"draws": draws, "seed": seed, "confidence_level": level,
                          "minimum_group_rows_per_year": min_rows,
                          "minimum_group_months_per_year": min_months,
                          "minimum_risk_and_reference_rows_per_bin": min_bin,
                          "neutral_tolerance": NEUTRAL_TOLERANCE,
                          "fixed_2024h2_bin_weights": True,
                          "bootstrap_month_universes": {key: list(value[2]) for key, value in periods.items()},
                          "undefined_denominator_policy": "unresolved interval; no redraw, dropping or conditional estimation",
                          "forecast_calendar_semantics": "origin <= event calendar date and event date < next origin; this is a calendar research scenario, not an exact issuance-time claim"},
        "missingness": _history_missingness(joined),
        "confounding": _persistence_confounding_report(source),
    })


def _favorite_gap(frame: pd.DataFrame, context_results: dict, draws: int, seed: int) -> dict[str, Any]:
    del context_results
    eligible = frame.loc[
        frame.date.ge("2024-07-01") & frame.date.lt("2026-01-01")
        & frame["_global_match"] & frame["_valid_global"]
        & (frame.p_c0 - .5).abs().gt(NEUTRAL_TOLERANCE)
    ].copy()
    if eligible.empty:
        return {"estimate": None, "interval": None, "status": "NO_ELIGIBLE_ROWS",
                "interpretation": "descriptive only; does not trigger fitting"}
    y = eligible.y.to_numpy(float)
    p = eligible.p_c0.to_numpy(float)
    favorite = np.maximum(p, 1.0 - p)
    won = np.where(p >= .5, y, 1.0 - y)
    favorite_gap = favorite - won
    eligible["_favorite_gap"] = favorite_gap
    blocks = eligible.date.dt.to_period("M").astype(str).to_numpy()
    interval = _monthly_interval(
        eligible["_favorite_gap"].to_numpy(float), blocks, draws=draws,
        seed=seed, confidence_level=.95,
        month_universe=DETECTION_MONTHS + REPLICATION_MONTHS,
    )
    return {"estimate": float(np.mean(favorite - won)), "interval": interval,
            "n": len(eligible), "status": "DESCRIPTIVE_ONLY",
            "interpretation": "favorite confidence minus favorite win indicator; descriptive only; does not trigger fitting"}


def _confounding_report(frame: pd.DataFrame) -> dict[str, Any]:
    primary = frame.loc[pd.to_datetime(frame.date).dt.year.eq(2026)].copy()
    report: dict[str, Any] = {"proxy_caveat": "Tournament-name/year blocks are proxies, not certified tournament editions.",
                              "support_by": {}}
    columns = [column for column in (
        "best_of", "competition_tier", "recipe_year", "calibration_year",
    ) if column in primary]
    primary["year"] = pd.to_datetime(primary.date).dt.year.astype(str)
    columns.append("year")
    if "unknown_players" in primary:
        unknown = pd.to_numeric(primary.unknown_players, errors="coerce")
        primary["source_unseen_support"] = np.where(unknown.isna(), "UNKNOWN", np.where(unknown.gt(0), "unseen", "seen"))
        columns.append("source_unseen_support")
    for column in columns:
        report["support_by"][column] = [
            {"level": _safe(key), "rows": int(len(group)),
             "unknown_rows": int(group[column].isna().sum())}
            for key, group in primary.groupby(column, dropna=False, sort=True)
        ]
    if "tournament_name" in primary:
        primary["tournament_name"] = primary.tournament_name.fillna("UNKNOWN").astype(str)
        primary["source_name_year_proxy"] = primary.tournament_name + "|" + primary.year
        report["source_name_year_proxy"] = [
            {"proxy": str(key), "rows": int(len(group))}
            for key, group in primary.groupby("source_name_year_proxy", sort=True)
        ]
    report["causal_claim"] = False
    report["turnover_claim"] = False
    return report


def _scores(frame: pd.DataFrame, probability_column: str) -> dict[str, Any]:
    y = frame.y.to_numpy(float)
    p = frame[probability_column].to_numpy(float)
    if len(y) == 0:
        return {"n": 0, "log_loss": None, "brier": None}
    ll = binary_log_loss_vector(y, p, epsilon=0.0)
    return {"n": int(len(y)), "log_loss": float(ll.mean()),
            "brier": float(np.square(p - y).mean())}


def _delta_result(
    frame: pd.DataFrame, left: str, right: str, *, draws: int, seed: int,
    confidence_level: float, brier_level: float = .95,
    month_universe: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    y = frame.y.to_numpy(float)
    a, b = frame[left].to_numpy(float), frame[right].to_numpy(float)
    ll_a = binary_log_loss_vector(y, a, epsilon=0.0)
    ll_b = binary_log_loss_vector(y, b, epsilon=0.0)
    ll_delta = ll_a - ll_b
    brier_delta = np.square(a - y) - np.square(b - y)
    months = pd.to_datetime(frame.date).dt.to_period("M").astype(str).to_numpy()
    ll_interval = _monthly_interval(ll_delta, months, draws=draws, seed=seed,
                                    confidence_level=confidence_level,
                                    month_universe=month_universe)
    brier_interval = _monthly_interval(brier_delta, months, draws=draws, seed=seed,
                                       confidence_level=brier_level,
                                       month_universe=month_universe)
    return {"candidate": left, "comparator": right, "n": int(len(frame)),
            "log_loss": ll_interval, "brier": brier_interval,
            "brier_nonworse": bool(brier_interval.get("delta") is not None
                                    and brier_interval["delta"] <= 0
                                    and brier_interval.get("ci_upper") is not None
                                    and brier_interval["ci_upper"] <= 0)}


def _equal_month_delta(frame: pd.DataFrame, left: str, right: str, draws: int, seed: int,
                       confidence_level: float) -> dict[str, Any]:
    y = frame.y.to_numpy(float)
    a, b = frame[left].to_numpy(float), frame[right].to_numpy(float)
    loss_a = binary_log_loss_vector(y, a, epsilon=0.0)
    loss_b = binary_log_loss_vector(y, b, epsilon=0.0)
    months = pd.to_datetime(frame.date).dt.to_period("M").astype(str)
    month_stats = pd.DataFrame({"month": months, "delta": loss_a - loss_b}).groupby("month").delta.mean()
    interval = paired_cluster_interval(month_stats.to_numpy(), month_stats.index.to_numpy(),
                                       draws=draws, seed=seed,
                                       confidence_level=confidence_level)
    return {**interval, "estimand": "equal-calendar-month mean log-loss difference"}


def _proxy_delta(frame: pd.DataFrame, left: str, right: str, draws: int, seed: int,
                 confidence_level: float) -> dict[str, Any]:
    y = frame.y.to_numpy(float)
    a, b = frame[left].to_numpy(float), frame[right].to_numpy(float)
    ll_a = binary_log_loss_vector(y, a, epsilon=0.0)
    ll_b = binary_log_loss_vector(y, b, epsilon=0.0)
    blocks = frame.tournament_name.fillna("UNKNOWN").astype(str) + "|" + pd.to_datetime(frame.date).dt.year.astype(str)
    result = paired_cluster_interval(ll_a - ll_b, blocks, draws=draws, seed=seed,
                                     confidence_level=confidence_level)
    return {**result, "estimand": "paired tournament-name/year proxy; not certified editions"}


def evaluate_context_predictions(
    frame: pd.DataFrame, config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score matched historical forecasts and evaluate the frozen context candidate."""
    cfg = dict(config or {})
    draws = int(cfg.get("draws", DRAWS))
    seed = int(cfg.get("seed", SEED))
    expected_total = int(cfg.get("expected_cohort_rows", 11550))
    expected_primary = int(cfg.get("expected_primary_rows", 3103))
    expected_warmup = int(cfg.get("expected_warmup_rows", 8447))
    primary_level = float(cfg.get("primary_confidence_level", EVALUATION_CONFIDENCE))
    guard_margin = float(cfg.get("guard_ll_margin", .005))
    required = {"golgg_match_id", "date", "label_release_day", "p_c0", "p_context", "p_global",
                "p_independent", "p_a0", "y", "best_of", "unknown_players", "tournament_name",
                "competition_tier"}
    _required(frame, required, "context prediction frame")
    if draws <= 0 or not 0 < primary_level < 1 or guard_margin < 0:
        raise ValueError("invalid context evaluation configuration")
    if len(frame) != expected_total:
        raise ValueError(f"matched cohort row count changed: expected {expected_total}, found {len(frame)}")
    work = frame.copy()
    if work.golgg_match_id.isna().any() or work.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("evaluation cohort must have unique nonmissing match identities")
    work["date"] = _dates(work.date, "date")
    work["label_release_day"] = _dates(work.label_release_day, "label_release_day")
    y = pd.to_numeric(work.y, errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("evaluation outcomes must be binary")
    work["y"] = y.astype(int)
    years = work.date.dt.year
    primary = years.eq(2026)
    warmup = years.lt(2026)
    if int(primary.sum()) != expected_primary or int(warmup.sum()) != expected_warmup or int((~(primary | warmup)).sum()):
        raise ValueError("matched cohort must contain exact 2026 primary rows and earlier C0 warmup rows")
    probability_columns = ("p_c0", "p_context", "p_global", "p_independent", "p_a0")
    for column in probability_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    _probabilities(work, ("p_c0",))
    if work.loc[primary, ["p_context", "p_global", "p_independent", "p_a0"]].isna().any().any():
        raise ValueError("2026 primary rows require context, global, independent and annual A0 forecasts")
    for column in ("p_context", "p_global", "p_independent", "p_a0"):
        primary_values = work.loc[primary, column].to_numpy(float)
        if not np.isfinite(primary_values).all() or ((primary_values <= 0) | (primary_values >= 1)).any():
            raise ValueError(f"{column} must be finite and strictly inside (0,1) on 2026 primary rows")
    if work.loc[warmup, "p_context"].isna().any() or not np.array_equal(
        work.loc[warmup, "p_context"].to_numpy(float), work.loc[warmup, "p_c0"].to_numpy(float),
    ):
        raise ValueError("all earlier delivered warmup forecasts must remain exact C0")
    evaluation = work.loc[primary].copy()
    observed_months = tuple(sorted(evaluation.date.dt.to_period("M").astype(str).unique()))
    primary_month_universe = tuple(cfg.get(
        "primary_month_universe",
        PRIMARY_MONTHS if expected_primary == 3103 else observed_months,
    ))
    score_frames = {"cohort": work, "primary_2026": evaluation}
    scores = {
        scope: {name: _scores(part, column) for name, column in {
            "c0": "p_c0", "context": "p_context", "global": "p_global",
            "independent": "p_independent", "annual_a0": "p_a0",
        }.items() if part[column].notna().all()}
        for scope, part in score_frames.items()
    }
    comparisons = {}
    for index, comparator in enumerate(("p_c0", "p_global", "p_independent")):
        label = {"p_c0": "context_minus_c0", "p_global": "context_minus_global",
                 "p_independent": "context_minus_independent"}[comparator]
        comparisons[label] = _delta_result(
            evaluation, "p_context", comparator, draws=draws, seed=seed,
            confidence_level=primary_level, month_universe=primary_month_universe,
        )
        comparisons[label]["equal_month_log_loss"] = _equal_month_delta(
            evaluation, "p_context", comparator, draws, seed, primary_level,
        )
        comparisons[label]["source_name_year_proxy_log_loss"] = _proxy_delta(
            evaluation, "p_context", comparator, draws, seed, primary_level,
        )
    major_tier = evaluation.competition_tier.astype("string").str.casefold()
    major_mask = major_tier.eq("major").fillna(False).to_numpy(dtype=bool)
    unknown = pd.to_numeric(evaluation.unknown_players, errors="coerce")
    masks = {
        "bo3": evaluation.best_of.to_numpy() == 3,
        "bo5": evaluation.best_of.to_numpy() == 5,
        "major": major_mask,
        "source_unseen_player": unknown.gt(0).fillna(False).to_numpy(),
    }
    guards: dict[str, Any] = {}
    for mask_name, mask in masks.items():
        subset = evaluation.loc[mask]
        for comparator_name, comparator in (("c0", "p_c0"), ("annual_a0", "p_a0")):
            key = f"{mask_name}_vs_{comparator_name}"
            if subset.empty:
                guards[key] = {"n": 0, "pass": False, "log_loss": None, "threshold": guard_margin}
                continue
            paired = _delta_result(subset, "p_context", comparator, draws=draws,
                                   seed=seed,
                                   confidence_level=.95,
                                   month_universe=primary_month_universe)
            upper = paired["log_loss"].get("ci_upper")
            guards[key] = {"n": len(subset), "pass": upper is not None and upper <= guard_margin,
                           "log_loss": paired["log_loss"], "threshold": guard_margin,
                           "candidate": "p_context", "comparator": comparator,
                           "unknown_support_rows": int(unknown.isna().sum()) if mask_name == "source_unseen_player" else None}
    neutral = (evaluation.p_c0 - .5).abs().le(NEUTRAL_TOLERANCE)
    neutral_probability_deviation = np.abs(evaluation.loc[neutral, "p_context"].to_numpy(float) - .5)
    neutral_stability = {
        "rows": int(neutral.sum()),
        "max_absolute_probability_deviation": float(neutral_probability_deviation.max())
        if len(neutral_probability_deviation) else None,
        "tolerance": NEUTRAL_TOLERANCE,
        "pass": bool(np.all(neutral_probability_deviation <= NEUTRAL_TOLERANCE)),
    }
    residual = evaluation.loc[~neutral].copy()
    residual_diagnostics = {
        "rows": len(residual), "neutral_rows_excluded": int(neutral.sum()),
        "favorite_residuals": _favorite_residual_diagnostics(residual),
        "neutral_tolerance": NEUTRAL_TOLERANCE,
        "scope": "neutral C0 rows excluded only from favorite/slope residual diagnostics; retained in proper scores and reports",
    }
    decisions = {
        "primary_log_loss_pass": all(
            comparison["log_loss"].get("ci_upper") is not None
            and comparison["log_loss"]["ci_upper"] < 0
            for comparison in comparisons.values()
        ),
        "primary_brier_pass": all(comparison["brier_nonworse"] for comparison in comparisons.values()),
        "eight_noninferiority_guards_pass": all(value["pass"] for value in guards.values()),
        "neutral_stability_pass": neutral_stability["pass"],
    }
    decisions["eligible_for_development_adoption"] = all(decisions.values())
    return _safe({
        "status": "COMPLETE", "cohort": {
            "rows": len(work), "primary_2026_rows": int(primary.sum()),
            "earlier_exact_c0_warmup_rows": int(warmup.sum()),
            "expected_rows": {"total": expected_total, "primary_2026": expected_primary, "warmup": expected_warmup},
            "coverage_pass": True,
        },
        "scores": scores, "primary_comparators": comparisons,
        "noninferiority_guards": guards, "neutral_rows": int(neutral.sum()),
        "neutral_stability": neutral_stability,
        "residual_diagnostics": residual_diagnostics,
        "history_token_support": _token_support(evaluation),
        "missingness": {
            "full_cohort_rows": len(work),
            "primary_rows": len(evaluation),
            "unknown_covariates_full_cohort": {
                column: int(work[column].isna().sum())
                for column in ("history_stat_coverage", "unknown_players", "history_token_fraction")
                if column in work
            },
            "unknown_context_flags_primary": {
                column: int(pd.to_numeric(evaluation[column], errors="coerce").isna().sum())
                for column in CONTEXT_COLUMNS if column in evaluation
            },
            "unknown_source_support_primary": int(unknown.isna().sum()),
        },
        "confounding": _confounding_report(work),
        "decision": decisions,
        "criteria": {
            "primary_log_loss": f"all three 98.333333% monthly upper bounds strictly below zero",
            "brier_nonworse": "point delta <= 0 AND 95% monthly bootstrap upper bound <= 0",
            "guard": f"all eight 95% monthly log-loss upper bounds <= {guard_margin}",
            "equal_month_sensitivity": "descriptive sensitivity, not certified editions",
            "source_name_year_proxy": "tournament-name/year proxy, not independently validated editions",
            "neutral_stability": "C0-neutral rows remain scored; candidate must stay within 1e-6 of 0.5",
            "headline_clipping": False,
        },
        "configuration": {"draws": draws, "seed": seed, "primary_confidence_level": primary_level,
                          "brier_confidence_level": .95, "guard_confidence_level": .95,
                          "guard_ll_margin": guard_margin,
                          "primary_month_universe": list(primary_month_universe)},
    })


def _favorite_residual_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    records = []
    if frame.empty:
        return {"bins": records}
    favorite = np.maximum(frame.p_c0.to_numpy(float), 1.0 - frame.p_c0.to_numpy(float))
    won = np.where(frame.p_c0.to_numpy(float) >= .5, frame.y.to_numpy(float), 1.0 - frame.y.to_numpy(float))
    bins = _bin_index(favorite)
    for index in range(5):
        selected = bins == index
        records.append({"low": float(CONFIDENCE_EDGES[index]), "high": float(CONFIDENCE_EDGES[index + 1]),
                        "rows": int(selected.sum()),
                        "favorite_minus_win_rate": float(np.mean(favorite[selected] - won[selected])) if selected.any() else None})
    return {"bins": records}


def _token_support(frame: pd.DataFrame) -> dict[str, Any]:
    if "history_token_fraction" not in frame:
        return {"rows": len(frame), "available_rows": 0, "missing_rows": len(frame),
                "mean": None, "median": None, "quantiles": None}
    values = pd.to_numeric(frame.history_token_fraction, errors="coerce")
    finite = values[np.isfinite(values.to_numpy(float))]
    return {
        "rows": len(frame), "available_rows": int(len(finite)),
        "missing_rows": int(values.isna().sum()),
        "mean": float(finite.mean()) if len(finite) else None,
        "median": float(finite.median()) if len(finite) else None,
        "quantiles": {
            str(level): float(finite.quantile(level)) if len(finite) else None
            for level in (.1, .5, .9)
        },
        "role": "mask-derived descriptive support only; not a fourth tested context",
    }
