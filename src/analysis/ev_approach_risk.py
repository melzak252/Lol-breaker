"""Chronological, finite-support risk diagnostics for frozen EV approaches.

All online feedback is admitted only after its effective label release time.
The methods are empirical/model-based diagnostics: none establishes a
per-match probability lower-bound guarantee on the historical cohorts.
"""
from __future__ import annotations

import hashlib
from bisect import insort
import heapq
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import betaincinv, expit, logit

from src.analysis.ev_approach_data import effective_times

METHODS = (
    "multiaccuracy",
    "multicalibration",
    "selective_risk",
    "conformal",
    "adaptive_conformal",
    "sequential_residual",
    "bayesian",
    "robust_kelly",
)
EPSILON = 1e-12
MIN_GROUP_SUPPORT = 20
RISK_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
CONFIDENCE_CUTS = (0.60, 0.70, 0.80)
CONFIDENCE_LABELS = ("neutral:0.50", "(0.50,0.60)", "[0.60,0.70)", "[0.70,0.80)", "[0.80,1.00]")


def _probabilities(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _confidence_group(favorite_probability: float) -> int:
    if favorite_probability == 0.5:
        return 0
    return 1 + int(np.searchsorted(CONFIDENCE_CUTS, favorite_probability, side="right"))


def _confidence_groups(probabilities: np.ndarray) -> np.ndarray:
    favorite = np.maximum(probabilities, 1.0 - probabilities)
    return np.asarray([_confidence_group(float(value)) for value in favorite], dtype=int)


def _side_orientation(probability: float) -> float:
    """Orient toward a strict favorite; an exact tie has no side orientation."""
    if probability > 0.5:
        return 1.0
    if probability < 0.5:
        return -1.0
    return 0.0


def _calibration_orientation(probability: float, market_probability: float) -> float:
    if probability > 0.5:
        return 1.0
    if probability < 0.5:
        return -1.0
    if np.isfinite(market_probability) and market_probability > 0.5:
        return 1.0
    if np.isfinite(market_probability) and market_probability < 0.5:
        return -1.0
    return 0.0


def _valid_probability(value: float) -> bool:
    return bool(np.isfinite(value) and 0.0 <= value <= 1.0)


def _effective_time(frame: pd.DataFrame, column: str) -> pd.Series:
    # The shared owner is responsible for scenario-only timestamp fallback;
    # actual columns remain untouched in the caller's frame.
    return pd.to_datetime(effective_times(frame, column), errors="coerce", utc=True)


def _audit_groups(
    probabilities: np.ndarray,
    market: np.ndarray,
    group_ids: np.ndarray,
    method: str,
) -> list[tuple[str, np.ndarray]]:
    confidence = _confidence_groups(probabilities)
    if method == "multicalibration":
        return [(f"confidence:{CONFIDENCE_LABELS[int(g)]}", group_ids == g) for g in range(len(CONFIDENCE_LABELS))]

    absolute_gap = np.abs(probabilities - market)
    return [
        ("all", np.ones(len(probabilities), dtype=bool)),
        ("confidence:below_0.60", confidence <= 1),
        ("confidence:0.60_to_0.70", confidence == 2),
        ("confidence:at_least_0.70", confidence >= 3),
        ("market_gap:below_0.05", np.isfinite(absolute_gap) & (absolute_gap < 0.05)),
        ("market_gap:at_least_0.05", np.isfinite(absolute_gap) & (absolute_gap >= 0.05)),
    ]


def _row_auditor_names(probability: float, market_probability: float, method: str) -> list[str]:
    favorite_group = _confidence_group(max(probability, 1.0 - probability))
    groups = _audit_groups(
        np.asarray([probability]),
        np.asarray([market_probability]),
        np.asarray([favorite_group], dtype=int),
        method,
    )
    return [name for name, mask in groups if bool(mask[0])]


def _update_auditor_statistics(
    statistics: dict[str, dict[str, float]],
    probability: float,
    market_probability: float,
    target: float,
    method: str,
) -> None:
    orientation = _calibration_orientation(probability, market_probability)
    if orientation == 0.0:
        return
    favorite_probability = max(probability, 1.0 - probability)
    favorite_target = target if orientation > 0.0 else 1.0 - target
    residual = favorite_target - favorite_probability
    curvature = favorite_probability * (1.0 - favorite_probability)
    for name in _row_auditor_names(probability, market_probability, method):
        group = statistics.setdefault(name, {"residual_sum": 0.0, "curvature_sum": 0.0, "support": 0.0})
        group["residual_sum"] += residual
        group["curvature_sum"] += curvature
        group["support"] += 1.0


def _current_auditor_correction(
    statistics: dict[str, dict[str, float]],
    probability: float,
    market_probability: float,
    method: str,
) -> tuple[float, int, int, list[str], dict[str, int]]:
    current_groups = _row_auditor_names(probability, market_probability, method)
    all_group_names = [
        name for name, _ in _audit_groups(
            np.asarray([probability]),
            np.asarray([market_probability]),
            np.asarray([_confidence_group(max(probability, 1.0 - probability))], dtype=int),
            method,
        )
    ]
    group_counts = {name: int(statistics.get(name, {}).get("support", 0)) for name in all_group_names}
    supported = [name for name in current_groups if group_counts.get(name, 0) >= MIN_GROUP_SUPPORT]
    overall_support = group_counts.get("all", max(group_counts.values(), default=0))
    if not supported:
        return 0.0, overall_support, 0, [], group_counts

    # Each auditor's fixed three-pass damped Newton offset uses only cumulative
    # past residual and curvature sufficient statistics, not a repeated batch fit.
    offsets = []
    for name in supported:
        group = statistics[name]
        target_offset = float(np.clip(
            group["residual_sum"] / (group["curvature_sum"] + 20.0),
            -0.25,
            0.25,
        ))
        offset = 0.0
        for _ in range(3):
            offset += 0.25 * (target_offset - offset)
        offsets.append(offset)
    orientation = _calibration_orientation(probability, market_probability)
    local_support = min((group_counts[name] for name in supported), default=0)
    return orientation * float(np.mean(offsets)), overall_support, local_support, supported, group_counts


def _conformal_quantile(scores: list[float], alpha: float) -> float:
    """Finite-sample split-conformal order statistic; infinity gives the full set."""
    n = len(scores)
    if n == 0:
        return math.inf
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return math.inf
    return float(scores[rank - 1])


def _receipt_spec(method: str, alpha: float) -> dict[str, Any]:
    common = {
        "method": method,
        "alpha": float(alpha),
        "lower_probability_guarantee": False,
        "chronology": "Only rows with effective label_available_at strictly earlier than the effective decision_at update a forecast.",
        "scenario_timing": "Scenario timestamps are accepted only through ev_approach_data.effective_times; actual timestamp columns are not backfilled.",
        "scope": "Each (cohort, horizon) is replayed separately; missing released support is unavailable/abstain.",
        "allow_semantics": "Finite calibrators and the grouped Bayesian model expose both sides to downstream EV selection once available; selective_risk approves only a threshold-qualified favorite; robust_kelly exposes both sides when group support exists, while its fixed-12%-tax quarter-Kelly fraction is a primary-tax diagnostic only and does not gate alternate tax sensitivities; conformal and sequential diagnostics do not approve bets.",
        "evaluation_only_fields": "Prediction-set membership and coverage flags use realized y only as post-outcome diagnostics; current-row outcomes never affect forecasts or policy permissions.",
    }
    specs = {
        "multiaccuracy": {
            "auditors": ["all", "favorite_probability <0.60", "0.60<=favorite_probability<0.70", "favorite_probability>=0.70", "abs(p_s-q)<0.05", "abs(p_s-q)>=0.05"],
            "support": {"overall_min_rows": MIN_GROUP_SUPPORT, "per_auditor_min_rows": MIN_GROUP_SUPPORT},
            "updates": {"fixed_iterations": 3, "step_size": 0.25, "ridge_pseudoinformation": 20, "logit_increment_clip": [-0.25, 0.25]},
            "assumptions": "Empirical finite-auditor calibration only; side-symmetric favored-side logistic residual updates. Fully neutral p_s=q=0.5 rows have no oriented residual and do not add fitting support. No generalization, conditional calibration, or guarantee beyond the declared groups is claimed.",
        },
        "multicalibration": {
            "groups": list(CONFIDENCE_LABELS),
            "support": {"overall_min_rows": MIN_GROUP_SUPPORT, "per_score_group_min_rows": MIN_GROUP_SUPPORT},
            "updates": {"fixed_iterations": 3, "step_size": 0.25, "ridge_pseudoinformation": 20, "logit_increment_clip": [-0.25, 0.25]},
            "assumptions": "Calibration is checked only in the five frozen side-symmetric favored-probability bins (an exact-neutral bin plus four confidence bins); fully neutral p_s=q=0.5 rows do not add fitting support. Finite support and historical drift limit generalization.",
        },
        "selective_risk": {
            "threshold_family": list(RISK_THRESHOLDS),
            "selection": "Choose the lowest threshold whose prior selected-side error plus one-sided Hoeffding diagnostic radius is <= alpha; each threshold needs 20 selected released rows.",
            "assumptions": "The Hoeffding radius is a diagnostic motivated by independent bounded Bernoulli errors; serial dependence, adaptive search, and temporal drift invalidate a formal risk guarantee. Thresholding is based only on prior released outcomes.",
        },
        "conformal": {
            "score": "Binary 1 - probability assigned to the realized outcome; calibration scores use each historical row's frozen p_s.",
            "quantile": "ceil((n+1)*(1-alpha))-th ascending score; rank>n yields the full {A,B} set.",
            "support": {"minimum_released_calibration_rows": MIN_GROUP_SUPPORT},
            "assumptions": "Split-conformal marginal set coverage requires exchangeability within the evaluation scope; chronology alone does not establish it. Set coverage is not a probability lower bound.",
        },
        "adaptive_conformal": {
            "score": "Binary 1 - probability assigned to the realized outcome; feedback enters by effective label-release timestamp.",
            "initial_target_alpha": float(alpha),
            "step_size": 0.02,
            "target_alpha_clip": [0.01, 0.50],
            "simultaneous_release_rule": "Rows sharing an effective label-release timestamp update alpha once using their mean set coverage.",
            "support": {"minimum_released_calibration_rows": MIN_GROUP_SUPPORT},
            "assumptions": "The delayed-feedback ACI target update is a long-run coverage controller, not a per-match probability bound; its coverage interpretation depends on the online protocol and is not established under arbitrary drift/dependence.",
        },
        "sequential_residual": {
            "residual": "y - p_s, bounded in [-1,1]; evidence is accumulated in effective label-release order.",
            "support": {"minimum_released_rows_for_diagnostic": MIN_GROUP_SUPPORT},
            "confidence_sequence": "Two-sided union-Hoeffding boundary sqrt(2*log(pi^2*n^2/(3*alpha))/n), spending delta_n=6*alpha/(pi^2*n^2) at each release count.",
            "assumptions": "Interpretation requires baseline probabilities predictable before outcomes, conditionally mean-zero residuals under the null, and bounded residuals; none is certified by historical data.",
        },
        "bayesian": {
            "groups": list(CONFIDENCE_LABELS),
            "prior": {"family": "Beta-Binomial", "alpha": 2.0, "beta": 2.0, "shared_by_group": True},
            "interval": "Equal-tailed model-based posterior interval with tail probability alpha/2; report side-specific lower and upper endpoints.",
            "support": {"minimum_released_rows_per_group": MIN_GROUP_SUPPORT},
            "assumptions": "Bernoulli outcomes in a fixed favorite-probability group with a Beta(2,2) prior; posterior intervals are model-assumed group uncertainty, not calibrated per-match guarantees.",
        },
        "robust_kelly": {
            "groups": list(CONFIDENCE_LABELS),
            "ambiguity_interval": "Two-sided Hoeffding interval for each fixed group's favored-side Bernoulli win rate, centered at released empirical frequency with radius sqrt(log(2/alpha)/(2*n)); clipped to [0,1]. The emitted p is the same oriented group empirical frequency; baseline p_s is retained separately.",
            "support": {"minimum_released_rows_per_group": MIN_GROUP_SUPPORT},
            "kelly": {"primary_tax_diagnostic_rate": 0.12, "fraction_of_full_kelly": 0.25, "max_free_capital_fraction": 0.025, "net_decimal_multiplier": "odds*(1-primary_tax_diagnostic_rate)", "diagnostic_only": True, "used_as_allow_mask": False},
            "assumptions": "The ambiguity interval requires independent Bernoulli draws within fixed groups for its nominal Hoeffding interpretation; group bounds are model/population diagnostics, not per-match guarantees or validated Kelly safety bounds. Exact-neutral rows have no identifiable favorite and do not contribute support.",
        },
    }
    return {**common, **specs[method]}


def chronological_risk_predictions(
    frame: pd.DataFrame,
    method: str,
    alpha: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay one frozen risk method per cohort/horizon using released history only.

    Returned ``p`` is the method's coherent A-side mean (or the untouched
    baseline for methods that only emit a set, diagnostic, or ambiguity bounds).
    ``p_low_a`` and ``p_low_b`` are nullable side-specific diagnostics; neither
    their existence nor ``allow_a``/``allow_b`` certifies a true lower bound.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown risk method {method!r}; expected one of {METHODS}")
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and strictly between zero and one")
    required = {"row_id", "cohort", "horizon", "decision_at", "label_available_at", "y", "p_s"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing risk input columns: {sorted(missing)}")
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise ValueError("Risk input row_id values must be present and unique")
    receipt = _receipt_spec(method, float(alpha))

    working = frame.reset_index(drop=True).copy()
    working["__risk_position"] = np.arange(len(working), dtype=int)
    effective_decision = _effective_time(working, "decision_at")
    effective_release = _effective_time(working, "label_available_at")
    baseline = _probabilities(working, "p_s")
    market = _probabilities(working, "q")
    targets = pd.to_numeric(working["y"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    valid_targets = np.isfinite(targets) & np.isin(targets, (0.0, 1.0))
    baseline_valid = np.asarray([_valid_probability(p) for p in baseline], dtype=bool)
    clipped = baseline_valid & ((baseline <= EPSILON) | (baseline >= 1.0 - EPSILON))
    baseline = np.where(baseline_valid, np.clip(baseline, EPSILON, 1.0 - EPSILON), np.nan)
    market = np.where(np.isfinite(market) & (market >= 0.0) & (market <= 1.0), market, np.nan)

    scope_columns = ["cohort", "horizon"]
    scopes = (
        [
            part["__risk_position"].to_numpy(dtype=int)
            for _, part in working.groupby(scope_columns, sort=False, dropna=False)
        ]
        if len(working)
        else []
    )
    metadata_columns = tuple(
        column for column in working.columns
        if column in {"golgg_match_id", "cohort", "horizon", "information_status"}
        or column.startswith("scenario_")
    )

    odds_a = _probabilities(working, "odds_a")
    odds_b = _probabilities(working, "odds_b")
    records: list[dict[str, Any] | None] = [None] * len(working)
    scope_receipts: list[dict[str, Any]] = []

    for positions in scopes:
        positions = positions.tolist()
        order = sorted(
            positions,
            key=lambda i: (
                pd.isna(effective_decision.iloc[i]),
                effective_decision.iloc[i].value if pd.notna(effective_decision.iloc[i]) else 0,
                str(working.iloc[i]["row_id"]),
            ),
        )
        pending: list[tuple[int, int]] = []
        released: list[int] = []
        released_scores: list[float] = []
        adaptive_alpha = float(alpha)
        adaptive_updates = 0
        prediction_set_covered: dict[int, bool] = {}
        available_count = 0
        invalid_time_count = 0
        auditor_statistics: dict[str, dict[str, float]] = {}
        risk_counts = {threshold: 0 for threshold in RISK_THRESHOLDS}
        risk_errors = {threshold: 0.0 for threshold in RISK_THRESHOLDS}
        favorite_group_counts = np.zeros(len(CONFIDENCE_LABELS), dtype=int)
        favorite_group_successes = np.zeros(len(CONFIDENCE_LABELS), dtype=float)
        released_residual_sum = 0.0
        feedback_digest = hashlib.sha256()
        feedback_count = 0
        max_feedback_release_at = pd.NaT

        for current in order:
            origin = effective_decision.iloc[current]
            if pd.isna(origin):
                invalid_time_count += 1
                origin = None
            if origin is not None:
                while pending and pending[0][0] < origin.value:
                    release_tick = pending[0][0]
                    released_batch = []
                    while pending and pending[0][0] == release_tick:
                        _, released_index = heapq.heappop(pending)
                        released_batch.append(released_index)
                    batch_coverage = []
                    for released_index in released_batch:
                        y_released = targets[released_index]
                        p_released = baseline[released_index]
                        if not valid_targets[released_index] or not np.isfinite(p_released):
                            continue
                        released.append(released_index)
                        released_residual_sum += y_released - p_released
                        feedback_id = str(working.iloc[released_index]["row_id"]).encode("utf-8")
                        feedback_digest.update(len(feedback_id).to_bytes(8, "big", signed=False))
                        feedback_digest.update(feedback_id)
                        feedback_count += 1
                        max_feedback_release_at = effective_release.iloc[released_index]
                        if method in ("multiaccuracy", "multicalibration"):
                            _update_auditor_statistics(
                                auditor_statistics, p_released, market[released_index], y_released, method,
                            )
                        favorite_probability_released = max(p_released, 1.0 - p_released)
                        favorite_group_released = _confidence_group(favorite_probability_released)
                        if method == "selective_risk" and p_released != 0.5:
                            error = float((y_released if p_released >= 0.5 else 1.0 - y_released) == 0.0)
                            for threshold in RISK_THRESHOLDS:
                                if favorite_probability_released >= threshold:
                                    risk_counts[threshold] += 1
                                    risk_errors[threshold] += error
                        if method in ("bayesian", "robust_kelly") and p_released != 0.5:
                            favorite_group_counts[favorite_group_released] += 1
                            favorite_group_successes[favorite_group_released] += (
                                y_released if p_released > 0.5 else 1.0 - y_released
                            )
                        realized_probability = p_released if y_released == 1.0 else 1.0 - p_released
                        insort(released_scores, float(1.0 - realized_probability))
                        if method == "adaptive_conformal":
                            batch_coverage.append(float(prediction_set_covered.get(released_index, False)))
                    if batch_coverage:
                        covered_fraction = float(np.mean(batch_coverage))
                        adaptive_alpha = float(np.clip(
                            adaptive_alpha + 0.02 * (float(alpha) - (1.0 - covered_fraction)),
                            0.01,
                            0.50,
                        ))
                        adaptive_updates += 1
            n_history = len(released)
            p_a = float(baseline[current]) if np.isfinite(baseline[current]) else np.nan
            record: dict[str, Any] = {
                "row_id": working.iloc[current]["row_id"],
                "method": method,
                "decision_at": working.iloc[current]["decision_at"],
                "label_available_at": working.iloc[current]["label_available_at"],
                "effective_decision_at": effective_decision.iloc[current],
                "effective_label_available_at": effective_release.iloc[current],
                "p_s": working.iloc[current]["p_s"],
                "p": p_a,
                **{column: working.iloc[current][column] for column in metadata_columns},
                "status": "unavailable",
                "missing_reason": None,
                "p_low_a": np.nan,
                "p_low_b": np.nan,
                "allow_a": False,
                "allow_b": False,
                "p_high_a": np.nan,
                "p_high_b": np.nan,
                "feedback_count": feedback_count,
                "feedback_ids_sha256": feedback_digest.copy().hexdigest(),
                "max_feedback_release_at": max_feedback_release_at,
                "support_count": n_history,
                "risk_group": None,
                "risk_threshold": np.nan,
                "risk_estimate": np.nan,
                "risk_threshold_family": None,
                "score_group": None,
                "abstention_reason": None,
                "risk_upper_diagnostic": np.nan,
                "risk_selected_support": 0,
                "set_contains_a": False,
                "set_contains_b": False,
                "prediction_set_size": 0,
                "prediction_set_covered": pd.NA,
                "conformal_quantile": np.nan,
                "conformal_support": len(released_scores),
                "adaptive_alpha": adaptive_alpha if method == "adaptive_conformal" else np.nan,
                "adaptive_updates": adaptive_updates if method == "adaptive_conformal" else 0,
                "residual_mean": np.nan,
                "residual_boundary": np.nan,
                "residual_lower": np.nan,
                "residual_upper": np.nan,
                "residual_alarm": False,
                "posterior_successes": np.nan,
                "posterior_failures": np.nan,
                "posterior_alpha": np.nan,
                "posterior_beta": np.nan,
                "ambiguity_lower_favorite": np.nan,
                "ambiguity_upper_favorite": np.nan,
                "robust_kelly_fraction_a": np.nan,
                "robust_kelly_fraction_b": np.nan,
                "auditor_correction": np.nan,
                "auditor_groups_used": [],
            }
            if origin is None:
                record["missing_reason"] = "unknown_effective_decision_time"
                record["abstention_reason"] = record["missing_reason"]
                records[current] = record
                continue
            if not np.isfinite(p_a):
                record["missing_reason"] = "missing_or_invalid_baseline_probability"
                record["abstention_reason"] = record["missing_reason"]
                records[current] = record
                continue

            favorite_probability = max(p_a, 1.0 - p_a)
            favorite_group = _confidence_group(favorite_probability)
            record["risk_group"] = CONFIDENCE_LABELS[favorite_group]
            record["score_group"] = CONFIDENCE_LABELS[favorite_group]
            available = False

            if method in ("multiaccuracy", "multicalibration"):
                correction, support, _, used, group_support = _current_auditor_correction(
                    auditor_statistics, p_a, market[current], method,
                )
                record["support_count"] = support
                record["auditor_correction"] = correction
                record["auditor_groups_used"] = used
                orientation = _calibration_orientation(p_a, market[current])
                if method == "multicalibration":
                    group_name = f"confidence:{CONFIDENCE_LABELS[favorite_group]}"
                    local_support = group_support.get(group_name, 0)
                    record["support_count"] = local_support
                    available = local_support >= MIN_GROUP_SUPPORT
                else:
                    available = support >= MIN_GROUP_SUPPORT and bool(used)
                if available:
                    favorite_p = max(p_a, 1.0 - p_a)
                    favorite_mean = float(expit(
                        logit(np.clip(favorite_p, EPSILON, 1.0 - EPSILON)) + correction * orientation
                    ))
                    record["p"] = favorite_mean if orientation > 0 else 1.0 - favorite_mean
                    record["missing_reason"] = None
                elif method == "multicalibration":
                    record["missing_reason"] = "insufficient_released_score_group_support"
                else:
                    record["missing_reason"] = "insufficient_released_auditor_support"
            elif method == "selective_risk":
                candidates: list[tuple[float, float, float, int]] = []
                for threshold in RISK_THRESHOLDS:
                    count = risk_counts[threshold]
                    if count < MIN_GROUP_SUPPORT:
                        continue
                    estimate = risk_errors[threshold] / count
                    radius = math.sqrt(math.log(1.0 / float(alpha)) / (2.0 * count))
                    upper = min(1.0, estimate + radius)
                    candidates.append((threshold, estimate, upper, count))
                acceptable = [candidate for candidate in candidates if candidate[2] <= float(alpha)]
                if acceptable:
                    threshold, estimate, upper, count = min(acceptable, key=lambda item: item[0])
                    orientation = _side_orientation(p_a)
                    if favorite_probability >= threshold and p_a != 0.5:
                        record["allow_a"] = orientation > 0
                        record["allow_b"] = orientation < 0
                    record["risk_threshold"] = threshold
                    record["risk_estimate"] = estimate
                    record["risk_upper_diagnostic"] = upper
                    record["risk_selected_support"] = count
                    available = True
                elif n_history >= MIN_GROUP_SUPPORT:
                    record["missing_reason"] = "no_threshold_passed_prior_risk_diagnostic"
                else:
                    record["missing_reason"] = "insufficient_released_risk_support"
                record["risk_threshold_family"] = list(RISK_THRESHOLDS)

            elif method in ("conformal", "adaptive_conformal"):
                active_alpha = adaptive_alpha if method == "adaptive_conformal" else float(alpha)
                quantile = _conformal_quantile(released_scores, active_alpha)
                contains_a = bool(1.0 - p_a <= quantile)
                contains_b = bool(p_a <= quantile)
                record["set_contains_a"] = contains_a
                record["set_contains_b"] = contains_b
                record["prediction_set_size"] = int(contains_a) + int(contains_b)
                record["prediction_set_covered"] = (
                    contains_a if valid_targets[current] and targets[current] == 1.0
                    else contains_b if valid_targets[current] else pd.NA
                )
                record["conformal_quantile"] = quantile
                record["conformal_support"] = len(released_scores)
                if method == "adaptive_conformal":
                    record["adaptive_alpha"] = active_alpha
                    record["adaptive_updates"] = adaptive_updates
                available = len(released_scores) >= MIN_GROUP_SUPPORT
                if not available:
                    record["missing_reason"] = "insufficient_released_conformal_support"
                prediction_set_covered[current] = bool(record["prediction_set_covered"]) if valid_targets[current] else False

            elif method == "sequential_residual":
                count = n_history
                if count:
                    mean = released_residual_sum / count
                    boundary = math.sqrt(2.0 * math.log((math.pi ** 2) * (count ** 2) / (3.0 * float(alpha))) / count)
                    record["residual_mean"] = mean
                    record["residual_boundary"] = boundary
                    record["residual_lower"] = max(-1.0, mean - boundary)
                    record["residual_upper"] = min(1.0, mean + boundary)
                    record["residual_alarm"] = bool(abs(mean) > boundary)
                available = count >= MIN_GROUP_SUPPORT
                if not available:
                    record["missing_reason"] = "insufficient_released_residual_support"

            elif method in ("bayesian", "robust_kelly"):
                orientation = _side_orientation(p_a)
                group_count = int(favorite_group_counts[favorite_group]) if orientation != 0.0 else 0
                successes = float(favorite_group_successes[favorite_group]) if orientation != 0.0 else 0.0
                failures = float(group_count - successes)
                record["support_count"] = group_count
                record["posterior_successes"] = successes
                record["posterior_failures"] = failures
                if method == "bayesian":
                    prior_a = prior_b = 2.0
                    posterior_a = prior_a + successes
                    posterior_b = prior_b + failures
                    record["posterior_alpha"] = posterior_a
                    record["posterior_beta"] = posterior_b
                    available = orientation != 0.0 and group_count >= MIN_GROUP_SUPPORT
                    if available:
                        favorite_mean = posterior_a / (posterior_a + posterior_b)
                        lower_favorite = float(betaincinv(posterior_a, posterior_b, float(alpha) / 2.0))
                        upper_favorite = float(betaincinv(posterior_a, posterior_b, 1.0 - float(alpha) / 2.0))
                        record["p"] = favorite_mean if orientation > 0 else 1.0 - favorite_mean
                        record["p_low_a"] = lower_favorite if orientation > 0 else 1.0 - upper_favorite
                        record["p_high_a"] = upper_favorite if orientation > 0 else 1.0 - lower_favorite
                        record["p_low_b"] = 1.0 - record["p_high_a"]
                        record["p_high_b"] = 1.0 - record["p_low_a"]
                    else:
                        record["missing_reason"] = "insufficient_released_bayesian_group_support"
                else:
                    available = orientation != 0.0 and group_count >= MIN_GROUP_SUPPORT
                    if available:
                        empirical = successes / group_count
                        record["p"] = empirical if orientation > 0.0 else 1.0 - empirical
                        radius = math.sqrt(math.log(2.0 / float(alpha)) / (2.0 * group_count))
                        lower_favorite = max(0.0, empirical - radius)
                        upper_favorite = min(1.0, empirical + radius)
                        record["ambiguity_lower_favorite"] = lower_favorite
                        record["ambiguity_upper_favorite"] = upper_favorite
                        record["p_low_a"] = lower_favorite if orientation > 0 else 1.0 - upper_favorite
                        record["p_high_a"] = upper_favorite if orientation > 0 else 1.0 - lower_favorite
                        record["p_low_b"] = 1.0 - record["p_high_a"]
                        record["p_high_b"] = 1.0 - record["p_low_a"]
                        odds_a_value = odds_a[current]
                        odds_b_value = odds_b[current]
                        for side, lower, odds in (
                            ("a", record["p_low_a"], odds_a_value),
                            ("b", record["p_low_b"], odds_b_value),
                        ):
                            fraction = 0.0
                            if np.isfinite(odds) and odds > 1.0:
                                net_multiplier = odds * (1.0 - 0.12)
                                net_profit = net_multiplier - 1.0
                                if net_profit > 0.0:
                                    full_kelly = max(0.0, (float(lower) * net_multiplier - 1.0) / net_profit)
                                    fraction = min(0.025, 0.25 * full_kelly)
                            record[f"robust_kelly_fraction_{side}"] = fraction
                    else:
                        record["missing_reason"] = "insufficient_released_ambiguity_group_support"

            if available:
                record["status"] = "available"
                if method in ("multiaccuracy", "multicalibration", "bayesian", "robust_kelly"):
                    record["allow_a"] = True
                    record["allow_b"] = True
                record["missing_reason"] = None
                available_count += 1
            else:
                if record["missing_reason"] is None:
                    record["missing_reason"] = "insufficient_released_support"
            record["abstention_reason"] = None if available else record["missing_reason"]
            records[current] = record

            release_at = effective_release.iloc[current]
            if pd.notna(release_at):
                heapq.heappush(pending, (release_at.value, current))

        scope_receipt = {"rows": len(positions), "available_rows": available_count,
                         "unavailable_rows": len(positions) - available_count,
                         "unknown_decision_time_rows": invalid_time_count}
        for column in scope_columns:
            values = working.iloc[positions][column].drop_duplicates().tolist()
            scope_receipt[column] = values[0] if len(values) == 1 else values
        scope_receipts.append(scope_receipt)

    output = pd.DataFrame([record for record in records if record is not None])
    # Preserve original order even with duplicate or non-monotonic source indices.
    output = output.reset_index(drop=True)
    available_rows = int(output["status"].eq("available").sum()) if len(output) else 0
    receipt.update({
        "rows": len(output),
        "available_rows": available_rows,
        "unavailable_rows": int(len(output) - available_rows),
        "abstentions": int((~(output["allow_a"] | output["allow_b"])).sum()) if len(output) else 0,
        "input_boundary_probability_clips": int(clipped.sum()),
        "groups_and_support_by_scope": scope_receipts,
        "output_columns": list(output.columns),
    })
    # A boolean nullable coverage diagnostic is deliberately not a prediction feature.
    if len(output):
        output["prediction_set_covered"] = pd.array(output["prediction_set_covered"], dtype="boolean")
    return output, receipt
