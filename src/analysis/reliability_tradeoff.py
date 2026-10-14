"""Historical reliability-first duration comparison; no fitting or promotion."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.a0_data_focus import _summary
from src.analysis.model_reliability import (
    CONFIDENCE_EDGES, NEUTRAL_TOLERANCE, _calibration, _profile_masks,
)
from src.analysis.production_market_benchmark import paired_cluster_interval
from src.analysis.research_benchmark import losses, safe, validate_frame
from src.models.a0_data_focus import MODES, PROJECT_ROOT, resolve_research_root
from src.models.a0_native_refit import _contract_digest, _sha256

VARIANTS = ("native12", "native6")
COMPONENTS = ("full", "neural", "raw_neural")
CRITICAL_PROFILES = ("bo3", "bo5", "major", "source_unseen_player")


def _probabilities(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.size == 0 or not np.isfinite(result).all() or np.any((result <= 0) | (result >= 1)):
        raise ValueError(f"{name} must be a nonempty finite probability array strictly inside (0, 1)")
    return result


def confidence_match_scale(candidate_cal, reference_cal) -> float:
    """Return the candidate/reference RMS-logit magnitude ratio on CAL."""
    candidate = _probabilities(candidate_cal, name="candidate_cal")
    reference = _probabilities(reference_cal, name="reference_cal")
    if candidate.shape != reference.shape:
        raise ValueError("CAL probability arrays must have matching nonempty shape")
    candidate_logits = np.log(candidate) - np.log1p(-candidate)
    reference_logits = np.log(reference) - np.log1p(-reference)
    denominator = float(np.mean(reference_logits * reference_logits))
    if denominator == 0:
        raise ValueError("reference CAL logits have zero RMS magnitude")
    numerator = float(np.mean(candidate_logits * candidate_logits))
    return float(np.sqrt(numerator / denominator))


def apply_confidence_scale(probabilities, scale) -> np.ndarray:
    """Apply a symmetric logit scale without clipping probabilities."""
    p = _probabilities(probabilities, name="probabilities")
    if (not isinstance(scale, (int, float, np.integer, np.floating))
            or not np.isfinite(scale) or scale < 0):
        raise ValueError("scale must be a finite nonnegative scalar")
    logits = np.log(p) - np.log1p(-p)
    scaled = float(scale) * logits
    # Stable logistic; the two branches avoid overflow for large finite logits.
    result = np.empty_like(scaled)
    positive = scaled >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-scaled[positive]))
    exponential = np.exp(scaled[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _calendar_cells(frame: pd.DataFrame, by_quarter: bool):
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("match dates must be present")
    anchor = np.asarray(frame["p_a0_full"], dtype=float)
    if not np.isfinite(anchor).all() or np.any((anchor < 0) | (anchor > 1)):
        raise ValueError("p_a0_full must contain finite probabilities in [0, 1]")
    neutral = np.abs(anchor - .5) <= NEUTRAL_TOLERANCE
    confidence = np.maximum(anchor, 1 - anchor)
    bin_index = np.searchsorted(CONFIDENCE_EDGES[1:], confidence, side="right")
    bin_index = np.minimum(bin_index, len(CONFIDENCE_EDGES) - 2)
    quarter = dates.dt.year.astype(str) + "-Q" + dates.dt.quarter.astype(str)
    labels = ([f"{q}|{b}" for q, b in zip(quarter, bin_index)] if by_quarter
              else [f"all|{b}" for b in bin_index])
    return dates, neutral, np.asarray(labels, dtype=object)


def calibration_contrast(frame, candidate, reference, *, resamples=5000, seed=20261001,
                         cluster="month", by_quarter=True) -> dict:
    """Compare fixed-A0-favorite residual RMS over common quarter/confidence cells.

    Each bootstrap replicate resamples whole calendar blocks, then weights each
    observed cell by its resampled population mass. Cells absent in a draw have
    zero mass and contribute zero; draws with no eligible rows are retained as
    undefined (not redrawn).
    """
    if cluster not in ("month", "tournament"):
        raise ValueError("cluster must be 'month' or 'tournament'")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    required = {"y", "date", "p_a0_full", candidate, reference}
    if not required.issubset(frame.columns):
        raise ValueError(f"frame is missing columns: {sorted(required - set(frame.columns))}")
    y_raw = frame.y.to_numpy()
    if not np.isin(y_raw, [0, 1]).all():
        raise ValueError("y must contain binary labels")
    y = y_raw.astype(int)
    for column in (candidate, reference):
        p = frame[column].to_numpy(dtype=float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f"{column} must contain finite probabilities in [0, 1]")
    dates, neutral, cell_labels = _calendar_cells(frame, by_quarter)
    keep = ~neutral
    n_neutral = int(neutral.sum())
    if cluster == "month":
        blocks = dates.dt.to_period("M").astype(str).to_numpy(dtype=object)
    else:
        if "tournament_name" not in frame:
            raise ValueError("tournament cluster requires tournament_name")
        names = frame.tournament_name.astype("string")
        missing_names = names.isna().to_numpy()
        year_strings = dates.dt.year.astype(str).to_numpy()
        blocks = np.asarray([
            f"UNKNOWN|{year_strings[i]}" if missing_names[i] else f"{names.iloc[i]}|{year_strings[i]}"
            for i in range(len(frame))
        ], dtype=object)
    full_block_count = len(np.unique(blocks))
    eligible_index = np.flatnonzero(keep)
    if not len(eligible_index):
        return {"n": 0, "neutral_excluded": n_neutral, "blocks": full_block_count,
                "cluster": cluster, "by_quarter": bool(by_quarter),
                "candidate": {"squared_gap": None, "rms_gap": None},
                "reference": {"squared_gap": None, "rms_gap": None},
                "delta_squared": {"point": None, "ci_low": None, "ci_high": None},
                "delta_rms": {"point": None, "ci_low": None, "ci_high": None},
                "cells": [],
                "support": {"zero_total_support_draws": int(resamples), "draws": int(resamples),
                            "valid_draws": 0, "undefined_draws": int(resamples),
                            "absent_cell_draws": 0, "absent_cell_instances": 0,
                            "missing_tournament_labels": (int(frame.tournament_name.isna().sum())
                                if cluster == "tournament" else 0), "cell_count": 0}}
    labels = cell_labels[eligible_index]
    cell_ids, cell_index = np.unique(labels, return_inverse=True)
    cell_count = len(cell_ids)
    # Anchor favorite direction is fixed; candidate orientation is never used.
    anchor = np.asarray(frame["p_a0_full"], float)
    favorite_candidate = np.where(anchor >= .5, np.asarray(frame[candidate], float), 1 - np.asarray(frame[candidate], float))
    favorite_reference = np.where(anchor >= .5, np.asarray(frame[reference], float), 1 - np.asarray(frame[reference], float))
    residual_candidate = favorite_candidate - np.where(anchor >= .5, y, 1-y)
    residual_reference = favorite_reference - np.where(anchor >= .5, y, 1-y)
    rc, rr = residual_candidate[eligible_index], residual_reference[eligible_index]
    observed_counts = np.bincount(cell_index, minlength=cell_count).astype(float)
    means_c = np.bincount(cell_index, weights=rc, minlength=cell_count) / observed_counts
    means_r = np.bincount(cell_index, weights=rr, minlength=cell_count) / observed_counts
    weights = observed_counts / observed_counts.sum()
    point_c = float(np.sum(weights * means_c ** 2))
    point_r = float(np.sum(weights * means_r ** 2))
    delta_point = point_c - point_r
    point_rms = float(np.sqrt(point_c) - np.sqrt(point_r))
    # Block x cell counts and residual sums allow all draws in a few matrix ops.
    block_labels, full_block_index = np.unique(blocks, return_inverse=True)
    block_index = full_block_index[eligible_index]
    block_cell_count = np.zeros((len(block_labels), cell_count), dtype=float)
    block_c = np.zeros_like(block_cell_count)
    block_r = np.zeros_like(block_cell_count)
    np.add.at(block_cell_count, (block_index, cell_index), 1)
    np.add.at(block_c, (block_index, cell_index), rc)
    np.add.at(block_r, (block_index, cell_index), rr)
    rng = np.random.default_rng(seed)
    multiplicity = np.zeros((resamples, len(block_labels)), dtype=np.int32)
    rows = np.arange(resamples)[:, None]
    np.add.at(multiplicity, (np.broadcast_to(rows, (resamples, len(block_labels))),
                             rng.integers(0, len(block_labels), size=(resamples, len(block_labels)))), 1)
    sampled_counts = multiplicity @ block_cell_count
    totals = sampled_counts.sum(axis=1)
    absent = int(np.any(sampled_counts == 0, axis=1).sum())
    absent_cell_instances = int(np.sum(sampled_counts == 0))
    sampled_c = multiplicity @ block_c
    sampled_r = multiplicity @ block_r
    with np.errstate(divide="ignore", invalid="ignore"):
        cell_c = sampled_c / sampled_counts
        cell_r = sampled_r / sampled_counts
        mass = sampled_counts / totals[:, None]
        boot_c = np.sum(np.where(sampled_counts > 0, mass * cell_c ** 2, 0), axis=1)
        boot_r = np.sum(np.where(sampled_counts > 0, mass * cell_r ** 2, 0), axis=1)
        boot_delta = boot_c - boot_r
        boot_rms = np.sqrt(boot_c) - np.sqrt(boot_r)
    zero_total = totals == 0
    boot_delta[zero_total] = np.nan
    boot_rms[zero_total] = np.nan
    valid = np.isfinite(boot_delta) & np.isfinite(boot_rms)
    eligible_block_count = int(np.count_nonzero(block_cell_count.sum(axis=1)))
    all_draws_defined = bool(valid.all()) and len(block_labels) >= 2 and eligible_block_count >= 2
    if all_draws_defined:
        delta_low, delta_high = np.quantile(boot_delta, [.025, .975])
        rms_low, rms_high = np.quantile(boot_rms, [.025, .975])
        delta_interval = {"point": delta_point, "ci_low": float(delta_low), "ci_high": float(delta_high)}
        rms_interval = {"point": point_rms, "ci_low": float(rms_low), "ci_high": float(rms_high)}
    else:
        delta_interval = {"point": delta_point, "ci_low": None, "ci_high": None}
        rms_interval = {"point": point_rms, "ci_low": None, "ci_high": None}
    cell_records = []
    for j, cell in enumerate(cell_ids):
        bin_id = int(str(cell).rsplit("|", 1)[1])
        lower, upper = CONFIDENCE_EDGES[bin_id], CONFIDENCE_EDGES[bin_id + 1]
        cell_records.append({"cell": str(cell), "quarter_confidence_cell": str(cell),
                             "n": int(observed_counts[j]),
                             "observed_blocks": int(np.count_nonzero(block_cell_count[:, j])),
                             "candidate_signed_gap": float(means_c[j]),
                             "reference_signed_gap": float(means_r[j]),
                             "candidate_squared_gap": float(means_c[j] ** 2),
                             "reference_squared_gap": float(means_r[j] ** 2),
                             "population_weight": float(weights[j])})
    return {"n": int(len(eligible_index)), "neutral_excluded": n_neutral,
            "blocks": full_block_count, "eligible_blocks": eligible_block_count,
            "cluster": cluster, "by_quarter": bool(by_quarter),
            "candidate": {"squared_gap": point_c, "rms_gap": float(np.sqrt(point_c))},
            "reference": {"squared_gap": point_r, "rms_gap": float(np.sqrt(point_r))},
            "delta_squared": delta_interval, "delta_rms": rms_interval,
            "cells": cell_records,
            "support": {"zero_total_support_draws": int(zero_total.sum()), "draws": int(resamples),
                        "valid_draws": int(valid.sum()), "undefined_draws": int((~valid).sum()),
                        "all_draws_defined": all_draws_defined,
                        "absent_cell_draws": absent, "absent_cell_instances": absent_cell_instances,
                        "missing_tournament_labels": (int(frame.tournament_name.isna().sum())
                            if cluster == "tournament" else 0),
                        "cell_count": int(cell_count)}}


def _selector_reliability_gate(frame, candidate, reference, config):
    """Apply all C0-relative gates on the released evidence available at this origin."""
    resamples, seed = int(config.get("resamples", 5000)), int(config.get("seed", 20261001))
    ll_margin = float(config.get("global_ll_ci_upper_max", .005))
    brier_margin = float(config.get("global_brier_ci_upper_max", .001))
    checks = {}
    for profile in ("all", *CRITICAL_PROFILES):
        mask = np.ones(len(frame), dtype=bool) if profile == "all" else _selector_profile_mask(frame, profile)
        if mask is None:
            checks[profile] = {"pass": False, "reason": "PROFILE_METADATA_UNAVAILABLE"}
            continue
        subset = frame.loc[mask]
        if subset.empty:
            checks[profile] = {"pass": False, "reason": "NO_EVIDENCE"}
            continue
        y_raw = subset.y.to_numpy()
        if not np.isin(y_raw, [0, 1]).all():
            raise ValueError("y must contain binary labels")
        y = y_raw.astype(int)
        months = pd.to_datetime(subset.date).dt.to_period("M").astype(str)
        ll_ci = paired_cluster_interval(losses(y, subset[candidate].to_numpy(float))
                                        - losses(y, subset[reference].to_numpy(float)),
                                        months, draws=resamples, seed=seed)
        a, b = subset[candidate].to_numpy(float), subset[reference].to_numpy(float)
        brier_ci = paired_cluster_interval((a-y)**2 - (b-y)**2, months,
                                           draws=resamples, seed=seed)
        checks[profile] = {"pass": _upper_below(ll_ci, ll_margin),
                           "log_loss": ll_ci, "brier": brier_ci}
        if profile == "all":
            reliability = calibration_contrast(subset, candidate, reference,
                                               resamples=resamples, seed=seed)
            checks[profile]["reliability"] = reliability
            checks[profile]["reliability_pass"] = bool(
                reliability["delta_rms"]["point"] is not None
                and reliability["delta_rms"]["point"] <= -.005
                and reliability["delta_squared"]["ci_high"] is not None
                and reliability["delta_squared"]["ci_high"] < 0)
            checks[profile]["pass"] = bool(checks[profile]["pass"]
                and _upper_below(brier_ci, brier_margin)
                and checks[profile]["reliability_pass"])
    optional_pairs = config.get("selector_optional_mode_pairs", {})
    optional_checks = {}
    for mode in ("no_w20", "no_organization"):
        pair = optional_pairs.get(mode)
        if not pair:
            optional_checks[mode] = {"pass": False, "reason": "MODE_COLUMNS_UNAVAILABLE"}
            continue
        y_raw = frame.y.to_numpy()
        if not np.isin(y_raw, [0, 1]).all():
            raise ValueError("y must contain binary labels")
        delta = losses(y_raw.astype(int), frame[pair[0]].to_numpy(float)) \
            - losses(y_raw.astype(int), frame[pair[1]].to_numpy(float))
        interval = paired_cluster_interval(delta, pd.to_datetime(frame.date).dt.to_period("M").astype(str),
                                           draws=resamples, seed=seed)
        optional_checks[mode] = {"pass": _upper_below(
            interval, float(config.get("optional_mode_ll_ci_upper_max", .005))),
            "log_loss": interval}
    temporal = []
    dates = pd.to_datetime(frame.date)
    for year in sorted(dates.dt.year.unique()):
        annual = frame.loc[dates.dt.year.eq(year)]
        result = calibration_contrast(annual, candidate, reference, resamples=resamples,
                                     seed=seed, by_quarter=True)
        delta = result["delta_rms"]["point"]
        temporal.append({"year": int(year), "n": result["n"],
                         "year_aggregate_rms_worsening": delta,
                         "year_aggregate_delta_rms_ci_low": result["delta_rms"]["ci_low"],
                         "year_aggregate_delta_rms_ci_high": result["delta_rms"]["ci_high"],
                         "quarter_diagnostics": result["cells"]})
    temporal_pass = bool(temporal and all(
        row["year_aggregate_rms_worsening"] is not None
        and row["year_aggregate_rms_worsening"] <= float(config.get("year_rms_worsening_max", .005))
        for row in temporal))
    absolute_checks = {}
    calibration_config = dict(config)
    calibration_config.setdefault("calibration_min_rows", 200)
    calibration_config.setdefault("calibration_min_months", 6)
    calibration_config.setdefault("calibration_gap_tolerance", .05)
    for profile in ("all", *CRITICAL_PROFILES):
        mask = np.ones(len(frame), bool) if profile == "all" else _selector_profile_mask(frame, profile)
        if mask is None or not np.any(mask):
            absolute_checks[profile] = {"status": "INSUFFICIENT_EVIDENCE"}
            continue
        absolute_checks[profile] = _calibration(frame.loc[mask], candidate, calibration_config)
    support_pass = all(item.get("status") == "SUPPORTED_IN_DEVELOPMENT"
                       for item in absolute_checks.values())
    return {"pass": bool(all(item.get("pass", False) for item in checks.values())
                         and all(item.get("pass", False) for item in optional_checks.values())
                         and temporal_pass and support_pass),
            "checks": checks, "optional_context": optional_checks,
            "temporal_stability": {"pass": temporal_pass, "years": temporal},
            "absolute_calibration_support": absolute_checks,
            "absolute_calibration_support_pass": support_pass}


def _selector_profile_mask(frame, profile):
    if profile in ("bo3", "bo5"):
        return frame.best_of.eq(int(profile[2:])).to_numpy() if "best_of" in frame else None
    if profile == "source_unseen_player":
        return frame.unknown_players.gt(0).to_numpy() if "unknown_players" in frame else None
    if profile == "major":
        if not {"tournament_name", "date"}.issubset(frame.columns):
            return None
        from src.models.competition_tiers import classify_competition
        return np.asarray([
            classify_competition(name, pd.Timestamp(day).date()).tier.value == "major"
            for name, day in zip(frame.tournament_name, frame.date)
        ], dtype=bool)
    raise ValueError(f"unknown selector profile {profile}")


def select_past_recipes(frame, candidate, reference, *, rule, config, years=(2024, 2025, 2026)) -> dict:
    """Select C0/C6 using strictly earlier, already-released assessment evidence."""
    if rule not in ("ll", "reliability"):
        raise ValueError("rule must be 'll' or 'reliability'")
    required = {"date", "label_release_day", "y", candidate, reference}
    if not required.issubset(frame.columns):
        raise ValueError(f"frame is missing columns: {sorted(required - set(frame.columns))}")
    dates = pd.to_datetime(frame.date, errors="raise")
    release = pd.to_datetime(frame.label_release_day, errors="raise")
    if dates.isna().any() or release.isna().any():
        raise ValueError("match and label release dates must be present")
    y_raw = frame.y.to_numpy()
    if not np.isin(y_raw, [0, 1]).all():
        raise ValueError("y must contain binary labels")
    y = y_raw.astype(int)
    for column in (candidate, reference):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
            raise ValueError(f"{column} must contain finite probabilities in (0, 1)")
    origins = {int(year): pd.Timestamp(int(year), 1, 1) for year in years}
    probs = np.asarray(frame[reference], dtype=float).copy()
    records = []
    if "golgg_match_id" in frame:
        if frame.golgg_match_id.isna().any() or frame.golgg_match_id.astype(str).duplicated().any():
            raise ValueError("match IDs must be nonnull and unique")
        ids = frame.golgg_match_id.astype(str).to_numpy()
    else:
        ids = frame.index.astype(str).to_numpy()
    for year in years:
        origin = origins[int(year)]
        eligible = (dates < origin) & (dates.dt.year < int(year)) & (release < origin)
        evidence = frame.loc[eligible]
        selected = reference
        gate_result = None
        ll_means = None
        if int(year) > int(min(years)) and not evidence.empty:
            labels = evidence.y.to_numpy(dtype=int)
            c_ll = float(losses(labels, evidence[candidate].to_numpy(float)).mean())
            r_ll = float(losses(labels, evidence[reference].to_numpy(float)).mean())
            ll_means = {candidate: c_ll, reference: r_ll}
            if rule == "ll":
                if c_ll < r_ll:
                    selected = candidate
            else:
                gate_result = _selector_reliability_gate(evidence, candidate, reference, config)
                if gate_result["pass"]:
                    selected = candidate
        output_mask = dates.dt.year.eq(int(year)).to_numpy()
        probs[output_mask] = frame.loc[output_mask, selected].to_numpy(float)
        eligible_ids = ids[np.asarray(eligible)]
        release_days = release.loc[eligible].dt.strftime("%Y-%m-%d").tolist()
        records.append({"year": int(year), "origin": origin.strftime("%Y-%m-%d"),
                        "selected_column": selected, "evidence_ids": eligible_ids.tolist(),
                        "evidence_count": int(len(eligible_ids)),
                        "release_cutoff_exclusive": origin.strftime("%Y-%m-%d"),
                        "max_evidence_release_day": max(release_days) if release_days else None,
                        "ll_mean": ll_means, "gates": gate_result})
    return {"probabilities": probs, "records": records, "rule": rule}


def _checked_json(path: Path):
    return json.loads(path.read_text())


def _canonical(protocol):
    canonical_path = PROJECT_ROOT / protocol["canonical_path"]
    if _sha256(canonical_path) != protocol["canonical_sha256"]:
        raise ValueError("canonical data hash differs from frozen protocol")
    base = pd.read_parquet(canonical_path)
    if "p_a0_full" not in base or "golgg_match_id" not in base or base.golgg_match_id.isna().any():
        raise ValueError("canonical data must include p_a0_full and nonnull IDs")
    base["golgg_match_id"] = base.golgg_match_id.astype(str)
    if len(base) != 11550 or base.golgg_match_id.duplicated().any():
        raise ValueError("exact unique canonical 11,550-row population is required")
    return base


def _validate_aligned(base, forecasts):
    forecasts = forecasts.copy()
    if "golgg_match_id" not in forecasts or forecasts.golgg_match_id.isna().any():
        raise ValueError("forecast IDs must be present and nonnull")
    forecasts["golgg_match_id"] = forecasts.golgg_match_id.astype(str)
    if forecasts.golgg_match_id.duplicated().any():
        raise ValueError("forecast IDs must be unique")
    if set(forecasts.golgg_match_id) != set(base.golgg_match_id):
        raise ValueError("forecast coverage differs from canonical IDs")
    aligned = forecasts.set_index("golgg_match_id").loc[base.golgg_match_id].reset_index()
    identity_fields = ("team1_id", "team2_id", "y", "date", "best_of", "roster_a", "roster_b")
    for key in identity_fields:
        if key not in base or key not in aligned:
            raise ValueError(f"forecast/canonical identity field is missing: {key}")
        left, right = base[key].reset_index(drop=True), aligned[key].reset_index(drop=True)
        if key == "date":
            left = pd.to_datetime(left).dt.strftime("%Y-%m-%d")
            right = pd.to_datetime(right).dt.strftime("%Y-%m-%d")
        elif key in ("team1_id", "team2_id", "y", "best_of"):
            left, right = left.astype(str), right.astype(str)
        if key in ("roster_a", "roster_b"):
            equal = all(tuple(map(str, a)) == tuple(map(str, b)) for a, b in zip(left, right))
        else:
            equal = left.equals(right)
        if not equal:
            raise ValueError(f"forecast {key} alignment differs from canonical")
    return aligned


def _verify_confidence_control(frame, endpoints, completion):
    required = {"partition", "model_year"}
    if not required.issubset(endpoints.columns):
        raise ValueError("endpoint confidence calibration membership is incomplete")
    if "model_year" not in frame:
        raise ValueError("assessment predictions lack model_year")
    frame_years = pd.to_numeric(frame.model_year, errors="raise")
    if not frame_years.astype(int).eq(pd.to_datetime(frame.date).dt.year).all():
        raise ValueError("assessment model_year differs from calendar year")
    audit = {}
    for year in (2024, 2025, 2026):
        cal = endpoints.loc[endpoints.partition.eq("CAL") & endpoints.model_year.eq(year)]
        if cal.empty:
            raise ValueError(f"missing confidence CAL rows for {year}")
        audit[str(year)] = {}
        for mode in MODES:
            scale = confidence_match_scale(cal[f"native6__full__{mode}"],
                                           cal[f"native12__full__{mode}"])
            recorded = completion["confidence_scales"][str(year)][mode]
            if not np.isclose(float(recorded), scale, rtol=0, atol=1e-12):
                raise ValueError(f"confidence scale receipt differs: {year}/{mode}")
            origin_scales = completion["origins"][str(year)]["calibration"]["scales"]
            if not np.isclose(float(origin_scales[mode]), scale, rtol=0, atol=1e-12):
                raise ValueError(f"origin confidence scale receipt differs: {year}/{mode}")
            selected = pd.to_numeric(frame.model_year, errors="raise").eq(year).to_numpy()
            expected = apply_confidence_scale(frame.loc[selected, f"native12__full__{mode}"], scale)
            observed = frame.loc[selected, f"confidence_matched__full__{mode}"].to_numpy(float)
            if not np.allclose(observed, expected, rtol=0, atol=1e-12):
                raise ValueError(f"assessment confidence transform differs: {year}/{mode}")
            audit[str(year)][mode] = {"scale": scale, "cal_rows": int(len(cal))}
    return audit


def evaluate_tradeoff(protocol_path, output_dir, research_root=None) -> dict:
    """Verify frozen study artifacts and write complete descriptive evaluation outputs."""
    protocol_path, output_dir = Path(protocol_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    protocol = _checked_json(protocol_path)
    if protocol["schema"] != "a0_reliability_tradeoff_v2":
        raise ValueError("unsupported reliability tradeoff protocol schema")
    run_id = protocol["run_id"]
    if Path(run_id).name != run_id or not run_id:
        raise ValueError("protocol run_id must be a single path component")
    root = resolve_research_root(research_root)
    source_hashes = protocol["source_sha256"]
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("frozen protocol must enumerate source hashes")
    for name, expected in source_hashes.items():
        source_path = root / name.removeprefix("research:") if name.startswith("research:") \
            else PROJECT_ROOT / name.removeprefix("project:")
        if _sha256(source_path) != expected:
            raise ValueError(f"frozen source checksum differs: {name}")
    snapshot_path = protocol["source_snapshot_path"]
    snapshot_sha = protocol["source_snapshot_sha256"]
    goal_snapshot_path = protocol["goal_snapshot_path"]
    goal_snapshot_sha = protocol["goal_snapshot_sha256"]
    if not all((snapshot_path, snapshot_sha, goal_snapshot_path, goal_snapshot_sha,
                protocol["initial_goal_path"], protocol["initial_goal_sha256"])):
        raise ValueError("protocol must bind source and immutable goal snapshots")
    if _sha256(PROJECT_ROOT / snapshot_path) != snapshot_sha:
        raise ValueError("source snapshot ZIP hash differs")
    if _sha256(PROJECT_ROOT / goal_snapshot_path) != goal_snapshot_sha:
        raise ValueError("immutable goal snapshot hash differs")
    with zipfile.ZipFile(PROJECT_ROOT / snapshot_path) as source_zip:
        archived_goal = source_zip.read("goal/initial_goal.md")
    if (hashlib.sha256(archived_goal).hexdigest() != goal_snapshot_sha
            or protocol["initial_goal_sha256"] != goal_snapshot_sha):
        raise ValueError("source ZIP or immutable Markdown goal snapshot differs")
    family_profile_path = protocol["family_profile_protocol_path"]
    family_profile_sha = protocol["family_profile_protocol_sha256"]
    with zipfile.ZipFile(PROJECT_ROOT / snapshot_path) as source_zip:
        archived_profiles = source_zip.read("project/" + family_profile_path)
    family_profile_file = PROJECT_ROOT / family_profile_path
    if (_sha256(family_profile_file) != family_profile_sha
            or hashlib.sha256(archived_profiles).hexdigest() != family_profile_sha
            or json.loads(archived_profiles)["family_profiles"] != protocol["family_profiles"]):
        raise ValueError("Plan07 family profiles differ from frozen protocol")
    manifest_path = protocol["manifest_path"]
    manifest_sha = protocol["manifest_sha256"]
    manifest_input_path = protocol["manifest_input_path"]
    manifest_input_sha = protocol["manifest_input_sha256"]
    with zipfile.ZipFile(PROJECT_ROOT / snapshot_path) as source_zip:
        archived_manifest = source_zip.read("project/" + manifest_path)
        archived_manifest_input = source_zip.read("research/" + manifest_input_path)
    manifest_bytes = (PROJECT_ROOT / manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    if (hashlib.sha256(archived_manifest).hexdigest() != manifest_sha
            or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha
            or manifest["input"]["path"] != manifest_input_path
            or manifest["input"]["sha256"] != manifest_input_sha
            or hashlib.sha256(archived_manifest_input).hexdigest() != manifest_input_sha
            or _sha256(root / manifest_input_path) != manifest_input_sha):
        raise ValueError("frozen research manifest/input provenance differs")
    reference_origins = protocol["reference_origins"]
    if set(reference_origins) != {"2024", "2025", "2026"}:
        raise ValueError("protocol must pin all Plan07 native12 reference origins")
    archived_trajectories = {}
    with zipfile.ZipFile(PROJECT_ROOT / snapshot_path) as source_zip:
        for origin in reference_origins.values():
            for trajectory in origin["first_phase_receipts"].values():
                archived_trajectories[trajectory["path"]] = source_zip.read(
                    "project/" + trajectory["path"])
    for year, origin in reference_origins.items():
        for path_key, hash_key in (("completed_path", "completed_sha256"),
                                   ("native12_output_path", "native12_output_sha256"),
                                   ("control_execution_path", "control_execution_sha256")):
            if _sha256(PROJECT_ROOT / origin[path_key]) != origin[hash_key]:
                raise ValueError(f"Plan07 native12 reference changed: {year}/{path_key}")
        from src.models.a0_reliability_tradeoff import _verify_redirected_control
        execution = _checked_json(PROJECT_ROOT / origin["control_execution_path"])
        if _verify_redirected_control(execution) != origin["redirected_completion"]:
            raise ValueError(f"Plan07 redirected control provenance changed: {year}")
        for relative, digest in origin["artifacts"].items():
            if _sha256(PROJECT_ROOT / relative) != digest:
                raise ValueError(f"Plan07 reference artifact changed: {relative}")
        for trajectory in origin["first_phase_receipts"].values():
            relative = trajectory["path"]
            if (_sha256(PROJECT_ROOT / relative) != trajectory["sha256"]
                    or hashlib.sha256(archived_trajectories[relative]).hexdigest()
                    != trajectory["sha256"]):
                raise ValueError(f"Plan07 trajectory receipt changed: {relative}")
    if protocol["evaluation"]["status"] != "NOT_RUN":
        raise ValueError("protocol evaluation status must remain NOT_RUN before evaluation")
    base = _canonical(protocol)
    completion_path = PROJECT_ROOT / "data/08_reporting/reliability_tradeoff" / run_id / "training_completed.json"
    completion = _checked_json(completion_path)
    if completion["schema"] != "a0_reliability_tradeoff_training_v1" \
            or completion["status"] != "TRAINING_COMPLETE_NOT_SCORED":
        raise ValueError("matching completed training receipt is required")
    if completion["protocol_path"] != str(protocol_path.relative_to(PROJECT_ROOT)):
        raise ValueError("training completion path is not bound to this protocol")
    if completion["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("training completion is not bound to this protocol")
    if completion["evaluation_status"] != "NOT_RUN":
        raise ValueError("training receipt evaluation status must be NOT_RUN")
    if set(completion["origins"]) != {"2024", "2025", "2026"}:
        raise ValueError("training receipt must include all three origins")
    if set(completion["memberships"]) != {"2024", "2025", "2026"}:
        raise ValueError("training receipt must bind all endpoint memberships")
    scales = completion["confidence_scales"]
    if not isinstance(scales, dict) or set(scales) != {"2024", "2025", "2026"}:
        raise ValueError("training receipt must bind confidence scales for all origins")
    prediction_path = PROJECT_ROOT / completion["predictions_path"]
    endpoint_path = PROJECT_ROOT / completion["endpoint_predictions_path"]
    artifacts = completion["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("training completion must bind all output artifacts")
    for relative, expected in artifacts.items():
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"training artifact checksum differs: {relative}")
    for path in (prediction_path, endpoint_path):
        relative = str(path.relative_to(PROJECT_ROOT))
        expected = artifacts.get(relative)
        if expected is None or _sha256(path) != expected:
            raise ValueError(f"training receipt does not bind output artifact: {path}")
    predictions = _validate_aligned(base, pd.read_parquet(prediction_path))
    endpoints = pd.read_parquet(endpoint_path)
    if "label_release_day" not in predictions or "model_year" not in predictions:
        raise ValueError("assessment predictions must contain release dates and model years")
    frame = base.copy()
    for column in predictions.columns:
        if column != "golgg_match_id" and column not in frame:
            frame[column] = predictions[column].to_numpy()
    probability_columns = []
    for variant in VARIANTS:
        for component in COMPONENTS:
            for mode in MODES:
                column = f"{variant}__{component}__{mode}"
                if column not in frame:
                    raise ValueError(f"missing required probability column: {column}")
                probability_columns.append(column)
    confidence_columns = [f"confidence_matched__full__{mode}" for mode in MODES]
    for column in confidence_columns:
        if column not in frame:
            raise ValueError(f"missing confidence-control column: {column}")
        probability_columns.append(column)
    validate_frame(frame, probability_columns)
    confidence_control = _verify_confidence_control(frame, endpoints, completion)
    masks = _profile_masks(frame, protocol["family_profiles"])
    config = dict(protocol["evaluation"])
    probability_columns.append("p_a0_full")
    fixed_config = {"status": "NOT_RUN", "resamples": 5000, "seed": 20261001,
                    "reliability_rms_improvement_min": .005,
                    "monthly_squared_gap_ci_upper_max": 0.,
                    "global_ll_ci_upper_max": .005, "global_brier_ci_upper_max": .001,
                    "critical_profile_ll_ci_upper_max": .005,
                    "optional_mode_ll_ci_upper_max": .005, "year_rms_worsening_max": .005,
                    "favorite_gap_tolerance": .05, "favorite_min_rows": 200, "favorite_min_months": 6,
                    "canonical_a0_reliability_rms_improvement_min": .005,
                    "canonical_a0_monthly_squared_gap_ci_upper_max": 0.,
                    "canonical_a0_global_ll_ci_upper_max": .005,
                    "canonical_a0_global_brier_ci_upper_max": .001,
                    "canonical_a0_critical_profile_ll_ci_upper_max": .005}
    if any(config.get(key) != value for key, value in fixed_config.items()):
        raise ValueError("evaluation config differs from the frozen design")
    required_profiles = ("all", "BO3", "BO5", "major", "source_unseen_player")
    if tuple(config["required_profiles"]) != required_profiles:
        raise ValueError("evaluation profile gates differ from the frozen design")
    config["calibration_min_rows"] = config["favorite_min_rows"]
    config["calibration_min_months"] = config["favorite_min_months"]
    config["calibration_gap_tolerance"] = config["favorite_gap_tolerance"]
    endpoints_report = _endpoint_report(endpoints, completion["memberships"])
    metric_rows, contrast_rows, calibration_rows, cell_rows = [], {}, {}, []
    for column in probability_columns:
        for profile, mask in masks.items():
            sub = frame.loc[mask]
            score = _summary(sub, column) if len(sub) else {"n": 0}
            metric_rows.append({"model": column, "profile": profile, **score})
    for mode in MODES:
        for component in COMPONENTS:
            candidate, reference = f"native6__{component}__{mode}", f"native12__{component}__{mode}"
            monthly = _contrast_tradeoff(frame, candidate, reference, config)
            contrast_rows[f"{candidate}_vs_{reference}"] = monthly
            if component == "full":
                contrast = calibration_contrast(frame, candidate, reference,
                    resamples=int(config["resamples"]), seed=int(config.get("seed", 20261001)))
                tournament = calibration_contrast(frame, candidate, reference,
                    resamples=int(config["resamples"]), seed=int(config.get("seed", 20261001)),
                    cluster="tournament")
                calibration_rows[candidate] = {"monthly": contrast, "tournament_year_proxy": tournament}
                for cell in contrast["cells"]:
                    cell_rows.append({"model": candidate, "mode": mode, **cell})
    for mode in MODES:
        candidate, reference = f"native6__full__{mode}", f"native12__full__{mode}"
        pooled = calibration_contrast(frame, candidate, reference, resamples=config["resamples"],
                                      seed=config["seed"], by_quarter=False)
        calibration_rows[candidate]["global_confidence_cells"] = pooled
        for cell in pooled["cells"]:
            cell_rows.append({"model": candidate, "mode": mode,
                              "summary": "global_fixed_confidence", **cell})
    # This CAL-fitted confidence transform is diagnostic, never a promotion candidate.
    confidence_mechanism = {}
    for mode in MODES:
        matched = f"confidence_matched__full__{mode}"
        for reference in (f"native12__full__{mode}", f"native6__full__{mode}"):
            name = f"{matched}_vs_{reference}"
            proper = _contrast_tradeoff(frame, matched, reference, config)
            rms = calibration_contrast(frame, matched, reference,
                resamples=config["resamples"], seed=config["seed"])
            tournament_rms = calibration_contrast(frame, matched, reference,
                resamples=config["resamples"], seed=config["seed"], cluster="tournament")
            contrast_rows[name] = proper
            confidence_mechanism[name] = {"proper_score_contrast": proper,
                "monthly_rms_contrast": rms, "tournament_year_proxy_rms_contrast": tournament_rms}
    primary = contrast_rows["native6__full__full_vs_native12__full__full"]
    primary_cal = calibration_rows["native6__full__full"]["monthly"]
    reliability_pass = (primary_cal["delta_rms"]["point"] is not None
                        and primary_cal["delta_rms"]["point"] <= -.005
                        and primary_cal["delta_squared"]["ci_high"] is not None
                        and primary_cal["delta_squared"]["ci_high"] < 0)
    primary_gates = {"reliability": bool(reliability_pass),
                     "global_log_loss": _upper_below(primary["log_loss_monthly"], .005),
                     "global_brier": _upper_below(primary["brier_monthly"], .001)}
    critical_contrasts = {}
    for profile in CRITICAL_PROFILES:
        entry = _contrast_for_mask(frame, "native6__full__full", "native12__full__full", masks[profile], config)
        critical_contrasts[profile] = entry
        primary_gates[f"critical_log_loss_{profile}"] = _upper_below(entry["log_loss_monthly"], .005)
    optional = {}
    for mode in ("no_w20", "no_organization"):
        c, r = f"native6__full__{mode}", f"native12__full__{mode}"
        optional[mode] = _upper_below(contrast_rows[f"{c}_vs_{r}"]["log_loss_monthly"], .005)
    primary_gates["optional_context"] = all(optional.values())
    quarter_year = _quarter_year_reliability(frame)
    primary_gates["temporal_stability"] = bool(quarter_year) and all(
        row["year_aggregate_rms_worsening"] is not None
        and row["year_aggregate_rms_worsening"] <= config["year_rms_worsening_max"]
        for row in quarter_year if row["quarter"] == 0)
    calibration_support = {}
    for profile in ("all", *CRITICAL_PROFILES):
        model_cal = _calibration(frame.loc[masks[profile]], "native6__full__full", config)
        calibration_support[profile] = model_cal
    support_pass = all(item.get("status") == "SUPPORTED_IN_DEVELOPMENT"
                       for item in calibration_support.values())
    primary_gates["absolute_calibration_support"] = support_pass
    canonical_replacement = _contrast_tradeoff(frame, "native6__full__full", "p_a0_full", config)
    canonical_reliability = calibration_contrast(frame, "native6__full__full", "p_a0_full",
        resamples=config["resamples"], seed=config["seed"])
    replacement = {"global_log_loss": _upper_below(canonical_replacement["log_loss_monthly"], .005),
                   "global_brier": _upper_below(canonical_replacement["brier_monthly"], .001),
                   "reliability": bool(canonical_reliability["delta_rms"]["point"] is not None
                       and canonical_reliability["delta_rms"]["point"] <= -.005
                       and canonical_reliability["delta_squared"]["ci_high"] is not None
                       and canonical_reliability["delta_squared"]["ci_high"] < 0),
                   "contrast": canonical_replacement, "reliability_contrast": canonical_reliability,
                   "primary_gates_pass": all(primary_gates.values())}
    for profile in CRITICAL_PROFILES:
        entry = _contrast_for_mask(frame, "native6__full__full", "p_a0_full", masks[profile], config)
        replacement[f"critical_log_loss_{profile}"] = _upper_below(entry["log_loss_monthly"], .005)
        replacement.setdefault("critical_contrasts", {})[profile] = entry
    replacement["eligible_for_historical_research_replacement"] = bool(
        replacement["primary_gates_pass"] and all(
            replacement[name] for name in ("global_log_loss", "global_brier", "reliability",
                                           *(f"critical_log_loss_{profile}" for profile in CRITICAL_PROFILES))))
    config["selector_optional_mode_pairs"] = {
        mode: (f"native6__full__{mode}", f"native12__full__{mode}")
        for mode in ("no_w20", "no_organization")}
    selections = {rule: select_past_recipes(frame, "native6__full__full", "native12__full__full",
                    rule=rule, config=config) for rule in ("ll", "reliability")}
    selection_realized = {}
    selection_policy_comparison = {}
    y_all = frame.y.to_numpy(int)
    month_all = pd.to_datetime(frame.date).dt.to_period("M").astype(str)
    for rule, selection in selections.items():
        selected_probability = selection["probabilities"]
        selected_losses = losses(y_all, selected_probability)
        selected_brier = (selected_probability - y_all) ** 2
        outcome_records = []
        for record in selection["records"]:
            mask = pd.to_datetime(frame.date).dt.year.eq(record["year"]).to_numpy()
            selected_frame = frame.loc[mask].copy()
            selected_frame["_selected_probability"] = selected_probability[mask]
            score = _summary(selected_frame, "_selected_probability")
            outcome_records.append({"year": record["year"], "n": int(mask.sum()),
                "selected_column": record["selected_column"], "log_loss": score.get("log_loss"),
                "brier_score": score.get("brier_score"),
                "native12_log_loss": _summary(selected_frame, "native12__full__full").get("log_loss"),
                "native6_log_loss": _summary(selected_frame, "native6__full__full").get("log_loss")})
        selection_realized[rule] = outcome_records
        comparison = {}
        for fixed_column in ("native12__full__full", "native6__full__full"):
            fixed_probability = frame[fixed_column].to_numpy(float)
            ll_difference = selected_losses - losses(y_all, fixed_probability)
            brier_difference = selected_brier - (fixed_probability - y_all) ** 2
            comparison[fixed_column] = {
                "log_loss_difference": paired_cluster_interval(ll_difference, month_all,
                    draws=config["resamples"], seed=config["seed"]),
                "brier_difference": paired_cluster_interval(brier_difference, month_all,
                    draws=config["resamples"], seed=config["seed"])}
        selection_policy_comparison[rule] = {
            "aggregate_realized": _summary(pd.DataFrame({"y": y_all, "policy": selected_probability}), "policy"),
            "paired_against_fixed_recipe": comparison}
    ll_policy = selections["ll"]["probabilities"]
    reliability_policy = selections["reliability"]["probabilities"]
    selection_policy_comparison["ll"]["paired_vs_reliability_rule"] = {
        "log_loss_difference": paired_cluster_interval(
            losses(y_all, ll_policy) - losses(y_all, reliability_policy), month_all,
            draws=config["resamples"], seed=config["seed"]),
        "brier_difference": paired_cluster_interval(
            (ll_policy-y_all)**2 - (reliability_policy-y_all)**2, month_all,
            draws=config["resamples"], seed=config["seed"])}
    for name, selection in selections.items():
        frame[f"selection_{name}"] = selection["probabilities"]
    favorite_direction_counts = {}
    neutral_transitions = {}
    confidence_ordering = {}
    assessment_years = pd.to_datetime(frame.date).dt.year
    for mode in MODES:
        c, r = f"native6__full__{mode}", f"native12__full__{mode}"
        candidate_probability = frame[c].to_numpy(float)
        reference_probability = frame[r].to_numpy(float)
        candidate_favorite, reference_favorite = candidate_probability > .5, reference_probability > .5
        candidate_neutral = np.abs(candidate_probability - .5) <= NEUTRAL_TOLERANCE
        reference_neutral = np.abs(reference_probability - .5) <= NEUTRAL_TOLERANCE
        favorite_direction_counts[mode] = {
            "different_favorite_directions": int(np.sum(candidate_favorite != reference_favorite)),
            "same_favorite_direction": int(np.sum(candidate_favorite == reference_favorite))}
        neutral_transitions[mode] = {
            "both_neutral": int(np.sum(candidate_neutral & reference_neutral)),
            "native6_only_neutral": int(np.sum(candidate_neutral & ~reference_neutral)),
            "native12_only_neutral": int(np.sum(~candidate_neutral & reference_neutral)),
            "neither_neutral": int(np.sum(~candidate_neutral & ~reference_neutral))}
        confidence_ordering[mode] = {}
        for year in (2024, 2025, 2026):
            mask = assessment_years.eq(year).to_numpy()
            year_frame = frame.loc[mask]
            confidence_ordering[mode][str(year)] = {
                column: _summary(year_frame, column).get("auc")
                for column in (f"native12__full__{mode}",
                               f"confidence_matched__full__{mode}",
                               f"native6__full__{mode}")}
    hindsight = min(("native6__full__full", "native12__full__full"),
                    key=lambda column: _summary(frame, column)["log_loss"])
    output_dir.mkdir(parents=True, exist_ok=False)
    joined_predictions_path = output_dir / "joined_predictions.parquet"
    frame.to_parquet(joined_predictions_path, index=False)
    endpoint_gaps = _endpoint_generalization_gaps(endpoints_report)
    profile_table = pd.DataFrame(metric_rows)
    profile_table.to_csv(output_dir / "profiles.csv", index=False)
    profile_table.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(output_dir / "reliability_cells.csv", index=False)
    pd.DataFrame(quarter_year).to_csv(output_dir / "quarter_year_reliability.csv", index=False)
    pd.DataFrame([
        {"model": model, "mode": model.rsplit("__", 1)[-1],
         "monthly_n": row["monthly"]["n"],
         "delta_rms": row["monthly"]["delta_rms"]["point"],
         "delta_squared": row["monthly"]["delta_squared"]["point"],
         "tournament_proxy_n": row["tournament_year_proxy"]["n"]}
        for model, row in calibration_rows.items()
    ]).to_csv(output_dir / "calibration.csv", index=False)
    pd.DataFrame([{"contrast": name, "candidate": value["candidate"], "reference": value["reference"],
                   "n": value["n"], **value["log_loss_monthly"]}
                  for name, value in contrast_rows.items()]).to_csv(output_dir / "contrasts.csv", index=False)
    pd.DataFrame([{"rule": rule, **record} for rule, selection in selections.items()
                  for record in selection["records"]]).to_csv(output_dir / "selection.csv", index=False)
    pd.DataFrame([{"rule": rule, **outcome} for rule, outcomes in selection_realized.items()
                  for outcome in outcomes]).to_csv(output_dir / "selection_realized.csv", index=False)
    result = {"schema": "reliability_tradeoff_evaluation_v1", "run_id": run_id,
              "scope": "EXPOSED_HISTORICAL_DEVELOPMENT_NOT_PROSPECTIVE",
              "rows": len(frame), "training_completion_sha256": _sha256(completion_path),
              "protocol_sha256": _sha256(protocol_path),
              "goal_provenance": {"path": protocol["initial_goal_path"],
                                  "sha256": protocol["initial_goal_sha256"],
                                  "snapshot_path": protocol["goal_snapshot_path"],
                                  "snapshot_sha256": protocol["goal_snapshot_sha256"]},
              "family_profile_provenance": {"path": family_profile_path, "sha256": family_profile_sha,
                                           "profiles": protocol["family_profiles"]},
              "research_manifest_provenance": {"path": manifest_path, "sha256": manifest_sha,
                                              "input_path": manifest_input_path,
                                              "input_sha256": manifest_input_sha},
              "source_snapshot": {"path": snapshot_path, "sha256": snapshot_sha},
              "plan07_reference_provenance": reference_origins,
              "source_sha256": source_hashes,
              "input_sha256": {"canonical": protocol["canonical_sha256"],
                               "predictions": _sha256(prediction_path), "endpoints": _sha256(endpoint_path),
                               "joined_scored_predictions": _sha256(joined_predictions_path)},
              "outputs": {"joined_scored_predictions": str(joined_predictions_path.relative_to(output_dir))},
              "coverage": {"canonical_rows": len(frame), "unique_ids": int(frame.golgg_match_id.nunique()),
                           "missing_probabilities": {column: int(frame[column].isna().sum())
                                                     for column in probability_columns}},
              "confidence_control_validation": confidence_control,
              "training_confidence_scales": completion["confidence_scales"],
              "primary_contrasts": contrast_rows, "reliability": calibration_rows,
              "gates": {"primary_and_critical": primary_gates, "critical_contrasts": critical_contrasts,
                        "optional_context": optional, "all_primary_gates_pass": all(primary_gates.values()),
                        "canonical_a0_replacement": replacement},
              "absolute_calibration_support": calibration_support,
              "year_quarter_reliability": quarter_year,
              "confidence_mechanism_contrasts": confidence_mechanism,
              "confidence_mechanism_ordering_auc_by_origin": confidence_ordering,
              "favorite_direction_counts": favorite_direction_counts,
              "neutral_transitions": neutral_transitions,
              "selection": {rule: {"rule": rule, "records": selection["records"]}
                            for rule, selection in selections.items()},
              "selection_realized_scores": selection_realized,
              "selection_policy_comparison": selection_policy_comparison,
              "hindsight_best_constant_recipe": hindsight,
              "hindsight_scores": _summary(frame, hindsight),
              "hindsight_is_optimistic": True,
              "endpoint_generalization_gaps": endpoint_gaps, "endpoints": endpoints_report,
              "limitations": ["Exposed historical development; no prospective confirmation.",
                  "Squared residual gaps include finite-sample variance and do not identify latent match probabilities.",
                  "Paired squared-gap intervals are in squared-residual units; RMS contrasts are in residual-probability units.",
                  "Tournament-name by year blocks are an uncertified edition proxy; missing labels use UNKNOWN|year.",
                  "Year aggregate point guards are not simultaneous confidence guarantees.",
                  "Pooled confidence-control metrics use origin-specific scales; rank invariance applies only within an origin/mode for positive scales.",
                  "TRAIN endpoints are in-sample and CAL endpoints are in-sample to calibration; neither is independent validation."]}
    (output_dir / "evaluation.json").write_text(json.dumps(safe(result), indent=2, sort_keys=True) + "\n")
    report_hashes = {path.name: _sha256(path) for path in output_dir.iterdir() if path.is_file()}
    (output_dir / "evaluation_completed.json").write_text(json.dumps({
        "schema": "reliability_tradeoff_evaluation_receipt_v1",
        "protocol_sha256": _sha256(protocol_path),
        "training_completion_sha256": _sha256(completion_path),
        "report_artifacts_sha256": report_hashes,
        "evaluation_sha256": _sha256(output_dir / "evaluation.json"),
    }, indent=2, sort_keys=True) + "\n")
    return safe(result)


def _contrast_tradeoff(frame, candidate, reference, config):
    return _contrast_for_mask(frame, candidate, reference, np.ones(len(frame), bool), config)


def _contrast_for_mask(frame, candidate, reference, mask, config):
    sub = frame.loc[np.asarray(mask, bool)]
    y = sub.y.to_numpy(int)
    a, b = sub[candidate].to_numpy(float), sub[reference].to_numpy(float)
    ll = losses(y, a) - losses(y, b)
    brier = (a-y)**2 - (b-y)**2
    date = pd.to_datetime(sub.date)
    month = date.dt.to_period("M").astype(str)
    names = sub.tournament_name.astype("string")
    missing_names = names.isna().to_numpy()
    proxy = np.asarray([
        f"UNKNOWN|{date.dt.year.iloc[i]}" if missing_names[i]
        else f"{names.iloc[i]}|{date.dt.year.iloc[i]}"
        for i in range(len(sub))
    ], dtype=object)
    kwargs = {"draws": int(config.get("resamples", 5000)), "seed": int(config.get("seed", 20261001))}
    return {"candidate": candidate, "reference": reference,
            "n": len(sub), "log_loss_monthly": paired_cluster_interval(ll, month, **kwargs),
            "brier_monthly": paired_cluster_interval(brier, month, **kwargs),
            "log_loss_tournament_year_proxy": paired_cluster_interval(ll, proxy, **kwargs),
            "brier_tournament_year_proxy": paired_cluster_interval(brier, proxy, **kwargs),
            "tournament_proxy_missing_names": int(missing_names.sum()),
            "tournament_proxy_missing_convention": "UNKNOWN|year"}


def _upper_below(interval, bound):
    upper = interval.get("ci_upper")
    return bool(interval.get("blocks", 0) >= 2 and upper is not None and np.isfinite(upper) and upper < bound)


def _quarter_year_reliability(frame):
    dates = pd.to_datetime(frame.date)
    rows = []
    for year in sorted(dates.dt.year.unique()):
        annual = frame.loc[dates.dt.year.eq(year)]
        annual_result = calibration_contrast(annual, "native6__full__full", "native12__full__full",
                                             resamples=5000, seed=20261001)
        rows.append({"year": int(year), "quarter": 0, "diagnostic_scope": "year_aggregate",
                     "n": annual_result["n"], "candidate_rms": annual_result["candidate"]["rms_gap"],
                     "reference_rms": annual_result["reference"]["rms_gap"],
                     "rms_worsening": annual_result["delta_rms"]["point"],
                     "delta_rms_ci_low": annual_result["delta_rms"]["ci_low"],
                     "delta_rms_ci_high": annual_result["delta_rms"]["ci_high"],
                     "year_aggregate_candidate_rms": annual_result["candidate"]["rms_gap"],
                     "year_aggregate_reference_rms": annual_result["reference"]["rms_gap"],
                     "year_aggregate_rms_worsening": annual_result["delta_rms"]["point"],
                     "year_aggregate_delta_squared": annual_result["delta_squared"]["point"],
                     "year_aggregate_delta_squared_ci_low": annual_result["delta_squared"]["ci_low"],
                     "year_aggregate_delta_squared_ci_high": annual_result["delta_squared"]["ci_high"],
                     "year_aggregate_delta_rms_ci_low": annual_result["delta_rms"]["ci_low"],
                     "year_aggregate_delta_rms_ci_high": annual_result["delta_rms"]["ci_high"]})
        annual_dates = pd.to_datetime(annual.date)
        for quarter in sorted(annual_dates.dt.quarter.unique()):
            quarter_frame = annual.loc[annual_dates.dt.quarter.eq(quarter)]
            result = calibration_contrast(quarter_frame, "native6__full__full", "native12__full__full",
                                          resamples=5000, seed=20261001, by_quarter=False)
            delta = result["delta_rms"]["point"]
            rows.append({"year": int(year), "quarter": int(quarter), "diagnostic_scope": "quarter_diagnostic",
                         "n": result["n"], "candidate_rms": result["candidate"]["rms_gap"],
                         "reference_rms": result["reference"]["rms_gap"],
                         "rms_worsening": delta, "delta_rms_ci_low": result["delta_rms"]["ci_low"],
                         "delta_rms_ci_high": result["delta_rms"]["ci_high"],
                         "year_aggregate_candidate_rms": annual_result["candidate"]["rms_gap"],
                         "year_aggregate_reference_rms": annual_result["reference"]["rms_gap"],
                         "year_aggregate_rms_worsening": annual_result["delta_rms"]["point"],
                         "year_aggregate_delta_squared": annual_result["delta_squared"]["point"],
                         "year_aggregate_delta_squared_ci_low": annual_result["delta_squared"]["ci_low"],
                         "year_aggregate_delta_squared_ci_high": annual_result["delta_squared"]["ci_high"],
                         "year_aggregate_delta_rms_ci_low": annual_result["delta_rms"]["ci_low"],
                         "year_aggregate_delta_rms_ci_high": annual_result["delta_rms"]["ci_high"]})
    return rows


def _endpoint_report(endpoints, memberships):
    if not isinstance(endpoints, pd.DataFrame) or endpoints.empty:
        raise ValueError("endpoint predictions must be complete and nonempty")
    required = {"partition", "y", "model_year", "golgg_match_id", "date", "best_of", "label_release_day"}
    if not required.issubset(endpoints.columns):
        raise ValueError("endpoint predictions lack required identity, partition, or release fields")
    if (not endpoints.partition.isin(("TRAIN", "CAL", "ASSESSMENT")).all()
            or endpoints.golgg_match_id.isna().any()
            or endpoints.y.isna().any() or not endpoints.y.isin([0, 1]).all()):
        raise ValueError("endpoint identity, partition, or labels are invalid")
    expected = {2024: {"TRAIN": 27705, "CAL": 1379, "ASSESSMENT": 4642},
                2025: {"TRAIN": 32625, "CAL": 1103, "ASSESSMENT": 3805},
                2026: {"TRAIN": 36205, "CAL": 1328, "ASSESSMENT": 3103}}
    report = {}
    for partition in ("TRAIN", "CAL", "ASSESSMENT"):
        subset = endpoints.loc[endpoints.partition.eq(partition)].copy()
        report[partition] = {"n": int(len(subset)),
                             "models": {}, "scoreline_nll": {}, "raw_scoreline_nll": {},
                             "scoreline_likelihood_scope": "ensemble diagnostic, not per-seed training objective",
                             "by_model_year": {}}
        for year, count in expected.items():
            member = subset.loc[subset.model_year.eq(year)]
            frozen = memberships[str(year)]
            key = {"TRAIN": "train", "CAL": "cal", "ASSESSMENT": "validation"}[partition]
            ids = member.golgg_match_id.astype(str).tolist()
            if (len(member) != count[partition] or len(ids) != len(set(ids))
                    or frozen["counts"][key] != count[partition]
                    or frozen["membership_sha256"][key] != _contract_digest(ids)):
                raise ValueError(f"incomplete or changed endpoint membership: {partition}/{year}")
            year_report = {"n": int(len(member)), "models": {}, "scoreline_nll": {},
                           "raw_scoreline_nll": {},
                           "scoreline_likelihood_scope": "ensemble diagnostic, not per-seed training objective"}
            for model in VARIANTS:
                for component in COMPONENTS:
                    for mode in MODES:
                        column = f"{model}__{component}__{mode}"
                        if column not in subset:
                            raise ValueError(f"endpoint output missing probability {column}")
                        values = subset[column].to_numpy(float)
                        if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
                            raise ValueError(f"invalid endpoint probability: {column}/{partition}")
                        if column not in report[partition]["models"]:
                            report[partition]["models"][column] = _summary(subset, column)
                        year_report["models"][column] = _summary(member, column)
                for prefix, output_name in (("scoreline_nll", "scoreline_nll"),
                                            ("raw_scoreline_nll", "raw_scoreline_nll")):
                    for mode in MODES:
                        column = f"{model}__{prefix}__{mode}"
                        if column not in subset:
                            raise ValueError(f"endpoint output missing ensemble scoreline NLL: {column}")
                        values = subset[column].to_numpy(float)
                        if not np.isfinite(values).all() or np.any(values < 0):
                            raise ValueError(f"invalid ensemble scoreline NLL: {column}")
                        if column not in report[partition][output_name]:
                            report[partition][output_name][column] = {
                                "n": int(len(values)), "mean": float(values.mean())}
                        year_values = member[column].to_numpy(float)
                        year_report[output_name][column] = {
                            "n": int(len(year_values)), "mean": float(year_values.mean())}
            report[partition]["by_model_year"][str(year)] = year_report
    return report


def _endpoint_generalization_gaps(report):
    result = {}
    for model, train in report["TRAIN"]["models"].items():
        cal = report["CAL"]["models"][model]
        assessment = report["ASSESSMENT"]["models"][model]
        result[model] = {
            "train_log_loss": train.get("log_loss"), "cal_log_loss": cal.get("log_loss"),
            "assessment_log_loss": assessment.get("log_loss"),
            "assessment_minus_train_log_loss": (
                assessment["log_loss"] - train["log_loss"]
                if assessment.get("log_loss") is not None and train.get("log_loss") is not None else None),
            "assessment_minus_cal_log_loss": (
                assessment["log_loss"] - cal["log_loss"]
                if assessment.get("log_loss") is not None and cal.get("log_loss") is not None else None),
            "assessment_minus_train_brier": (
                assessment["brier_score"] - train["brier_score"]
                if assessment.get("brier_score") is not None and train.get("brier_score") is not None else None),
            "assessment_minus_cal_brier": (
                assessment["brier_score"] - cal["brier_score"]
                if assessment.get("brier_score") is not None and cal.get("brier_score") is not None else None),
        }
        by_year = {}
        for year in report["TRAIN"]["by_model_year"]:
            train_year = report["TRAIN"]["by_model_year"][year]["models"][model]
            cal_year = report["CAL"]["by_model_year"][year]["models"][model]
            assessment_year = report["ASSESSMENT"]["by_model_year"][year]["models"][model]
            by_year[year] = {
                "train": train_year, "cal": cal_year, "assessment": assessment_year,
                "assessment_minus_train_log_loss": assessment_year["log_loss"] - train_year["log_loss"],
                "assessment_minus_cal_log_loss": assessment_year["log_loss"] - cal_year["log_loss"],
                "assessment_minus_train_brier": assessment_year["brier_score"] - train_year["brier_score"],
                "assessment_minus_cal_brier": assessment_year["brier_score"] - cal_year["brier_score"]}
        result[model]["by_model_year"] = by_year
    return result
