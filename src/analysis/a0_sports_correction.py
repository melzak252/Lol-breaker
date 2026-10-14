"""Pure evaluation for the frozen A0-anchored sports correction experiment."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analysis.model_benchmark import compute_benchmark_summary
from src.analysis.production_market_benchmark import paired_cluster_interval
from src.analysis.research_benchmark import losses
from src.models.competition_tiers import classify_competition


PLAN09_PROBABILITY_COLUMNS = (
    "p_a0", "p_c0", "p_specialist", "p_connector03",
    "p_c0_calibration", "p_c0_connector",
)


PROBABILITY_COLUMNS = ("p_a0", "p_specialist", "p_calibration", "p_correction")
PROFILE_DIMENSIONS = (
    "year", "best_of", "competition_tier", "competition_family", "experience",
    "coplay", "confidence", "disagreement", "unknown_players",
    "new_organization", "w20_min_gate",
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _metrics(part: pd.DataFrame, probability_columns=PROBABILITY_COLUMNS) -> dict[str, Any]:
    result: dict[str, Any] = {}
    y = part.y.to_numpy(dtype=int)
    for column in probability_columns:
        if part.empty:
            result[column] = {"sample_size": 0, "log_loss": None, "brier_score": None,
                              "auc": None, "accuracy": None, "ece": None, "mce": None,
                              "calibration_slope": None, "calibration_intercept": None}
            continue
        summary = compute_benchmark_summary(y, part[column].to_numpy(float), probability_epsilon=None)
        result[column] = _safe(summary.__dict__)
        result[column]["tail_losses_ge_2_5"] = int(np.sum(losses(y, part[column].to_numpy(float)) >= 2.5))
        confidence = np.maximum(part[column].to_numpy(float), 1 - part[column].to_numpy(float))
        high = confidence >= .9
        won = np.where(part[column].to_numpy(float) > .5, y, 1 - y)
        result[column]["high_confidence"] = {
            "n": int(high.sum()), "observed_upsets": int(np.sum(1 - won[high])),
            "expected_upsets": float(np.sum(1 - confidence[high])),
        }
    return result


def _interval(part: pd.DataFrame, left_loss: np.ndarray, right_loss: np.ndarray,
              evaluation: dict[str, Any], *, sensitivity: bool = False) -> dict[str, Any]:
    if part.empty:
        blocks = np.asarray([], dtype=object)
    elif sensitivity:
        blocks = (part.tournament_name.fillna("unknown").astype(str) + "|" +
                  pd.to_datetime(part.date).dt.year.astype(str)).to_numpy()
    else:
        blocks = pd.to_datetime(part.date).dt.strftime("%Y-%m").to_numpy()
    return _safe(paired_cluster_interval(
        left_loss - right_loss, blocks,
        draws=int(evaluation.get("resamples", 5000)),
        seed=int(evaluation.get("seed", 20261001)),
    ))


def _contrasts(part: pd.DataFrame, evaluation: dict[str, Any], *, sensitivity: bool = False) -> dict[str, Any]:
    y = part.y.to_numpy(dtype=int)
    names = {"candidate_minus_anchor": "p_a0", "candidate_minus_control": "p_calibration"}
    candidate = losses(y, part.p_correction.to_numpy(float)) if len(part) else np.asarray([], dtype=float)
    candidate_brier = (part.p_correction.to_numpy(float) - y) ** 2
    result = {}
    for label, comparator in names.items():
        if len(part):
            compare_loss = losses(y, part[comparator].to_numpy(float))
            compare_brier = (part[comparator].to_numpy(float) - y) ** 2
        else:
            compare_loss = compare_brier = np.asarray([], dtype=float)
        result[label] = {
            "log_loss": _interval(part, candidate, compare_loss, evaluation, sensitivity=sensitivity),
            "brier": _interval(part, candidate_brier, compare_brier, evaluation, sensitivity=sensitivity),
        }
    return result


def _favorite(part: pd.DataFrame, probability_columns=PROBABILITY_COLUMNS) -> dict[str, Any]:
    anchor = part.p_a0.to_numpy(float)
    neutral = np.isclose(anchor, .5, rtol=0., atol=1e-12)
    keep = ~neutral
    y = part.y.to_numpy(int)
    anchor_favorite_won = np.where(anchor > .5, y, 1 - y)
    result: dict[str, Any] = {
        "n": int(keep.sum()), "neutral_excluded": int(neutral.sum()),
        "own_favorite": {}, "fixed_a0_favorite": {},
        "gap_definition": "predicted_minus_observed",
    }
    for col in probability_columns:
        p = part[col].to_numpy(float)
        own_keep = ~np.isclose(p, .5, rtol=0., atol=1e-12)
        own_confidence = np.maximum(p, 1 - p)
        own_won = np.where(p > .5, y, 1 - y)
        fixed_probability = np.where(anchor > .5, p, 1 - p)
        for name, selected, predicted, observed, probability_name in (
            ("own_favorite", own_keep, own_confidence, own_won, "mean_predicted_confidence"),
            ("fixed_a0_favorite", keep, fixed_probability, anchor_favorite_won,
             "mean_probability_assigned_to_a0_favorite"),
        ):
            result[name][col] = {
                "n": int(selected.sum()), "neutral_excluded": int((~selected).sum()),
                probability_name: float(predicted[selected].mean()) if selected.any() else None,
                "observed_favorite_win_rate": float(observed[selected].mean()) if selected.any() else None,
                "gap": float(np.mean(predicted[selected] - observed[selected])) if selected.any() else None,
            }
    return result


def _favorite_cells(part: pd.DataFrame, probability_columns=PROBABILITY_COLUMNS) -> dict[str, Any]:
    if part.empty:
        return {"confidence_edges": [.5, .6, .7, .8, .9, 1.0], "cells": [], "rms_gap": {c: None for c in probability_columns}}
    anchor = part.p_a0.to_numpy(float)
    keep = ~np.isclose(anchor, .5, rtol=0., atol=1e-12)
    favored_side_won = np.where(anchor > .5, part.y.to_numpy(int), 1 - part.y.to_numpy(int))
    anchor_confidence = np.maximum(anchor, 1 - anchor)
    quarters = pd.to_datetime(part.date).dt.to_period("Q").astype(str).to_numpy()
    labels = np.asarray([".5-.6", ".6-.7", ".7-.8", ".8-.9", ".9-1"])
    cell_rows = []
    weighted_squared: dict[str, list[tuple[int, float]]] = {c: [] for c in probability_columns}
    bins = np.searchsorted([.6, .7, .8, .9], anchor_confidence, side="right")
    for quarter in sorted(set(quarters)):
        for bin_index, label in enumerate(labels):
            selected = keep & (quarters == quarter) & (bins == bin_index)
            count = int(selected.sum())
            row: dict[str, Any] = {"quarter": quarter, "confidence_cell": str(label), "n": count}
            observed = float(favored_side_won[selected].mean()) if count else None
            row["observed_fixed_a0_favorite_win_rate"] = observed
            for col in probability_columns:
                probability = part[col].to_numpy(float)
                confidence = np.where(anchor > .5, probability, 1 - probability)
                predicted = float(confidence[selected].mean()) if count else None
                gap = predicted - observed if count else None
                row[col] = {"mean_probability_assigned_to_a0_favorite": predicted, "gap": gap}
                if count:
                    weighted_squared[col].append((count, float(gap) ** 2))
            cell_rows.append(row)
    rms = {}
    for col, items in weighted_squared.items():
        total = sum(n for n, _ in items)
        rms[col] = float(np.sqrt(sum(n * sq for n, sq in items) / total)) if total else None
    return {"confidence_edges": [.5, .6, .7, .8, .9, 1.0], "cells": cell_rows, "rms_gap": rms}


def _major_mask(part: pd.DataFrame) -> np.ndarray:
    identities = [classify_competition(name, timestamp.date()).tier.value
                  for name, timestamp in zip(part.tournament_name, pd.to_datetime(part.date))]
    return np.asarray([tier == "major" for tier in identities], dtype=bool)


def _profile_summaries(part: pd.DataFrame, dimensions: list[str],
                       probability_columns=PROBABILITY_COLUMNS) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in dimensions:
        if dimension == "year":
            values = pd.to_datetime(part.date).dt.year.astype(str) if len(part) else pd.Series(dtype=str)
        elif dimension == "competition_tier":
            values = pd.Series([classify_competition(name, timestamp.date()).tier.value
                                for name, timestamp in zip(part.tournament_name, pd.to_datetime(part.date))], index=part.index)
        elif dimension == "competition_family":
            values = pd.Series([classify_competition(name, timestamp.date()).family
                                for name, timestamp in zip(part.tournament_name, pd.to_datetime(part.date))], index=part.index)
        elif dimension in part.columns:
            values = part[dimension].fillna("unknown").astype(str)
        else:
            result[dimension] = {"missing_source_dimension": True, "groups": {}}
            continue
        groups = {}
        for value, indexes in values.groupby(values, sort=True).groups.items():
            subset = part.loc[indexes]
            groups[str(value)] = {"n": len(subset), "metrics": _metrics(subset, probability_columns)}
        result[dimension] = {"missing_source_dimension": False, "groups": groups}
    return result


def evaluate_correction(frame: pd.DataFrame, protocol: dict) -> dict[str, Any]:
    """Evaluate aligned historical forecasts; perform no fitting, clipping, or I/O."""
    evaluation = protocol["evaluation"]
    required = {"golgg_match_id", "date", "result_day", "feature_source_max_day", "best_of", "y",
                "tournament_name", *PROBABILITY_COLUMNS}
    required.add("unknown_players")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.empty or frame.golgg_match_id.isna().any() or frame.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("empty frame or duplicate match identity")
    identity_columns = ["golgg_match_id", "date", "result_day", "feature_source_max_day", "best_of", "y"]
    if frame[identity_columns].isna().any().any():
        raise ValueError("null match identity, result, or date")
    date = pd.to_datetime(frame.date, errors="raise")
    result_day = pd.to_datetime(frame.result_day, errors="raise")
    source_day = pd.to_datetime(frame.feature_source_max_day, errors="raise")
    if not (source_day < date).all() or not (result_day >= date).all():
        raise ValueError("invalid chronology: history must precede date and result cannot precede date")
    if not frame.y.isin([0, 1]).all() or not frame.best_of.isin([1, 3, 5]).all():
        raise ValueError("invalid outcome or best_of")
    for column in PROBABILITY_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
            raise ValueError(f"invalid strict probability in {column}")
    expected = int(evaluation["expected_rows"])
    primary_expected = int(evaluation["primary_expected_rows"])
    start = pd.Timestamp(evaluation.get("primary_start", "2025-01-01"))
    primary = date >= start
    if len(frame) != expected:
        raise ValueError(f"expected_rows mismatch: expected {expected}, got {len(frame)}")
    if int(primary.sum()) != primary_expected:
        raise ValueError(f"primary_expected_rows mismatch: expected {primary_expected}, got {int(primary.sum())}")

    work = frame.copy().reset_index(drop=True)
    work["date"] = date.to_numpy()
    work["_year"] = date.dt.year.to_numpy()
    cohorts: dict[str, pd.DataFrame] = {
        "full": work,
        "post_warmup": work.loc[primary.to_numpy()],
        "year2025": work.loc[work._year == 2025],
        "year2026": work.loc[work._year == 2026],
    }
    metrics = {name: _metrics(part) for name, part in cohorts.items()}
    contrasts = {name: _contrasts(part, evaluation) for name, part in cohorts.items()}
    contrasts["post_warmup"]["source_tournament_year_sensitivity"] = _contrasts(
        cohorts["post_warmup"], evaluation, sensitivity=True)

    major = _major_mask(work)
    unseen_source = pd.to_numeric(work.unknown_players, errors="coerce")
    unseen = unseen_source.to_numpy() > 0
    guard_masks = {
        "bo3": work.best_of.to_numpy() == 3,
        "bo5": work.best_of.to_numpy() == 5,
        "major": major,
        "source_unseen_player": unseen,
    }
    margin = float(evaluation.get("guard_ll_margin", .005))
    guards = {}
    for name in evaluation.get("critical_guards", guard_masks):
        part = work.loc[guard_masks[name] & primary.to_numpy()]
        delta = _contrasts(part, evaluation)
        anchor = delta["candidate_minus_anchor"]["log_loss"]
        control = delta["candidate_minus_control"]["log_loss"]
        passed = (anchor["ci_upper"] is not None and control["ci_upper"] is not None and
                  anchor["ci_upper"] < margin and control["ci_upper"] < margin)
        guards[name] = {"n": len(part), "candidate_minus_anchor_log_loss": anchor,
                        "candidate_minus_control_log_loss": control, "margin": margin,
                        "unknown_profile_rows": int((unseen_source.isna().to_numpy() & primary.to_numpy()).sum())
                        if name == "source_unseen_player" else None,
                        "noninferior": bool(passed)}

    primary_contrasts = contrasts["post_warmup"]
    predictive = all(primary_contrasts[key]["log_loss"]["ci_upper"] is not None and
                     primary_contrasts[key]["log_loss"]["ci_upper"] < 0 and
                     primary_contrasts[key]["brier"]["delta"] is not None and
                     primary_contrasts[key]["brier"]["delta"] <= 0
                     for key in ("candidate_minus_anchor", "candidate_minus_control"))
    full_anchor = metrics["full"]["p_a0"]
    full_candidate = metrics["full"]["p_correction"]
    full_direction = (full_candidate["log_loss"] is not None and full_anchor["log_loss"] is not None and
                      full_candidate["brier_score"] is not None and full_anchor["brier_score"] is not None and
                      full_candidate["log_loss"] < full_anchor["log_loss"] and
                      full_candidate["brier_score"] < full_anchor["brier_score"])
    decision = {"predictive_success": bool(predictive), "full_cohort_ll_brier_improve_vs_a0": bool(full_direction),
                "critical_guards_pass": bool(guards) and all(item["noninferior"] for item in guards.values())}
    decision["eligible"] = bool(decision["predictive_success"] and decision["full_cohort_ll_brier_improve_vs_a0"] and decision["critical_guards_pass"])

    dimensions = list(evaluation.get("profiles", PROFILE_DIMENSIONS))
    profile_summaries = {name: _profile_summaries(part, dimensions) for name, part in
                         (("full", cohorts["full"]), ("post_warmup", cohorts["post_warmup"]))}
    favorite = {name: {"summary": _favorite(part), "fixed_a0_quarter_confidence": _favorite_cells(part)}
                for name, part in (("full", work), ("post_warmup", cohorts["post_warmup"]))}
    return _safe({
        "cohort": {"rows": len(work), "primary_rows": int(primary.sum()),
                   "years": {str(year): int((work._year == year).sum()) for year in sorted(work._year.unique())}},
        "metrics": metrics, "contrasts": contrasts, "guards": guards,
        "profiles": profile_summaries, "favorite_calibration": favorite,
        "decision": decision,
        "limitations": {"reliability_established": False, "prospective_evidence": False,
                        "independent_edition_evidence": False,
                        "note": "Exposed historical development only; sports scores alone do not establish reliability or executable EV."},
    })


def evaluate_c0(frame: pd.DataFrame, protocol: dict) -> dict[str, Any]:
    """Evaluate Plan09's explicit C0 bindings and primary/conditional gates."""
    from src.analysis.production_market_benchmark import paired_cluster_interval

    required = {"golgg_match_id", "date", "y", "best_of", "tournament_name", "unknown_players",
                *PLAN09_PROBABILITY_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required Plan09 columns: {missing}")
    if frame.empty or frame.golgg_match_id.isna().any() or frame.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("Plan09 cohort is empty or has duplicate identities")
    if len(frame) != protocol["cohort"]["rows"]:
        raise ValueError("Plan09 cohort row count changed")
    for column in PLAN09_PROBABILITY_COLUMNS:
        p = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
            raise ValueError(f"invalid strict probability in {column}")
    if not frame.y.isin([0, 1]).all() or not frame.best_of.isin([1, 3, 5]).all():
        raise ValueError("invalid Plan09 outcomes or series format")
    y = frame.y.to_numpy(int)
    date = pd.to_datetime(frame.date, errors="raise")
    primary = date.ge(protocol["cohort"]["primary_start"]) & date.lt(protocol["cohort"]["primary_end_exclusive"])
    if int(primary.sum()) != protocol["cohort"]["primary_rows"]:
        raise ValueError("Plan09 primary cohort changed")
    candidate = "p_c0_connector"
    comparators = ("p_c0", "p_c0_calibration", "p_connector03")
    if protocol["evaluation"]["primary_contrasts"] != list(comparators):
        raise ValueError("Plan09 requires all three explicit primary comparators")
    primary_frame = frame.loc[primary].copy()

    def contrast(part, reference, sensitivity=False):
        labels = (part.tournament_name.fillna("unknown").astype(str) + "|" +
                  pd.to_datetime(part.date).dt.year.astype(str)).to_numpy() if sensitivity else pd.to_datetime(part.date).dt.strftime("%Y-%m").to_numpy()
        local_y = part.y.to_numpy(int)
        pred = part[candidate].to_numpy(float)
        base = part[reference].to_numpy(float)
        delta_ll = losses(local_y, pred) - losses(local_y, base)
        delta_brier = (pred - local_y) ** 2 - (base - local_y) ** 2
        output = {"log_loss": paired_cluster_interval(delta_ll, labels, 5000, 20261002),
                  "brier": paired_cluster_interval(delta_brier, labels, 5000, 20261002)}
        for value in output.values():
            for key, item in value.items():
                if isinstance(item, (float, np.floating)) and not np.isfinite(item):
                    value[key] = None
        return output
    months = np.sort(pd.to_datetime(primary_frame.date).dt.strftime("%Y-%m").unique())
    shared_month_draws = np.random.default_rng(20261002).integers(0, len(months), size=(5000, len(months)))

    def conditional_contrast(part, reference):
        y_local = part.y.to_numpy(int)
        candidate_local = part.p_c0_connector.to_numpy(float)
        reference_local = part[reference].to_numpy(float)
        ll = losses(y_local, candidate_local) - losses(y_local, reference_local)
        brier = (candidate_local - y_local) ** 2 - (reference_local - y_local) ** 2
        row_month = pd.to_datetime(part.date).dt.strftime("%Y-%m").to_numpy()
        values = {}
        for label, delta in (("log_loss", ll), ("brier", brier)):
            sums = np.zeros(len(months))
            counts = np.zeros(len(months))
            for index, month in enumerate(months):
                selected = row_month == month
                sums[index] = delta[selected].sum()
                counts[index] = selected.sum()
            samples = shared_month_draws
            drawn_counts = counts[samples].sum(axis=1)
            zero_replicates = int(np.sum(drawn_counts == 0))
            weighted = np.divide(sums[samples].sum(axis=1), drawn_counts,
                                 out=np.full(5000, np.nan), where=drawn_counts > 0)
            month_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            support = counts > 0
            sampled_support = support[samples].sum(axis=1)
            equal = np.divide(month_means[samples].sum(axis=1), sampled_support,
                              out=np.full(5000, np.nan), where=sampled_support > 0)
            valid = drawn_counts > 0
            values[label] = {
                "n": len(part), "blocks": int(support.sum()),
                "delta": float(delta.mean()) if len(delta) else None,
                "ci_lower": float(np.quantile(weighted[valid], .025)) if valid.all() and len(weighted) else None,
                "ci_upper": float(np.quantile(weighted[valid], .975)) if valid.all() and len(weighted) else None,
                "equal_block_delta": float(month_means[support].mean()) if support.any() else None,
                "equal_block_ci_lower": float(np.quantile(equal, .025)) if not zero_replicates else None,
                "equal_block_ci_upper": float(np.quantile(equal, .975)) if not zero_replicates else None,
                "draws": 5000, "seed": 20261002, "zero_support_replicates": zero_replicates,
            }
        return values

    primary_frame = frame.loc[primary].copy()
    contrasts = {ref: contrast(primary_frame, ref) for ref in comparators}
    unseen_source = pd.to_numeric(primary_frame.unknown_players, errors="coerce")
    masks = {
        "best_of_3": primary_frame.best_of.eq(3),
        "best_of_5": primary_frame.best_of.eq(5),
        "major": np.asarray([classify_competition(name, timestamp.date()).tier.value == "major"
                             for name, timestamp in zip(primary_frame.tournament_name, pd.to_datetime(primary_frame.date))]),
        "source_unseen_player": unseen_source.gt(0),
    }
    guards = {}
    for name, mask in masks.items():
        subset = primary_frame.loc[np.asarray(mask, dtype=bool)]
        guards[name] = {
            "n": len(subset),
            "unknown_profile_rows": int(unseen_source.isna().sum()) if name == "source_unseen_player" else None,
            "comparators": {ref: conditional_contrast(subset, ref) for ref in ("p_c0", "p_a0")},
        }
    primary_pass = all(
        contrasts[ref]["log_loss"]["ci_upper"] is not None
        and contrasts[ref]["log_loss"]["ci_upper"] < 0
        and contrasts[ref]["brier"]["delta"] is not None
        and contrasts[ref]["brier"]["delta"] <= 0 for ref in comparators)
    guard_pass = all(item["n"] > 0 and all(
        result["log_loss"]["ci_upper"] is not None and result["log_loss"]["ci_upper"] < .005
        for result in item["comparators"].values()) for item in guards.values())
    warmup = date.lt(protocol["cohort"]["primary_start"])
    sensitivity = {ref: contrast(primary_frame, ref, sensitivity=True) for ref in comparators}
    warmup_frame = frame.loc[warmup].copy()
    cohorts = {"full": frame, "primary": primary_frame, "warmup": warmup_frame}
    metrics = {name: _metrics(part, PLAN09_PROBABILITY_COLUMNS) for name, part in cohorts.items()}
    dimensions = [name for name in PROFILE_DIMENSIONS if name != "w20_min_gate"] + ["w20", "head_origin"]
    profiles = {
        name: _profile_summaries(part, dimensions, PLAN09_PROBABILITY_COLUMNS)
        for name, part in (("full", frame), ("primary", primary_frame))
    }
    favorite = {
        name: {"summary": _favorite(part, PLAN09_PROBABILITY_COLUMNS),
               "fixed_a0_quarter_confidence": _favorite_cells(part, PLAN09_PROBABILITY_COLUMNS)}
        for name, part in cohorts.items()
    }
    warmup_disclosure = {
        "rows": len(warmup_frame),
        "connector_exact_c0": bool(np.array_equal(warmup_frame.p_c0_connector, warmup_frame.p_c0)),
        "calibration_exact_c0": bool(np.array_equal(warmup_frame.p_c0_calibration, warmup_frame.p_c0)),
        "connector03_exact_a0": bool(np.array_equal(warmup_frame.p_connector03, warmup_frame.p_a0)),
        "connector_minus_c0_log_loss": float((losses(warmup_frame.y.to_numpy(int), warmup_frame.p_c0_connector.to_numpy(float)) -
                                                losses(warmup_frame.y.to_numpy(int), warmup_frame.p_c0.to_numpy(float))).mean()),
        "c0_minus_a0_log_loss": float((losses(warmup_frame.y.to_numpy(int), warmup_frame.p_c0.to_numpy(float)) -
                                     losses(warmup_frame.y.to_numpy(int), warmup_frame.p_a0.to_numpy(float))).mean())
        if len(warmup_frame) else None,
        "c0_minus_a0_brier": float(((warmup_frame.p_c0.to_numpy(float) - warmup_frame.y.to_numpy(int)) ** 2 -
                                  (warmup_frame.p_a0.to_numpy(float) - warmup_frame.y.to_numpy(int)) ** 2).mean())
        if len(warmup_frame) else None,
    }
    unresolved = any(
        result["log_loss"]["zero_support_replicates"] > 0
        for item in guards.values() for result in item["comparators"].values()
    )
    return _safe({
        "schema": "plan09_c0_evaluation_v1",
        "classification": protocol.get("classification", "EXPOSED_HISTORICAL_DEVELOPMENT"),
        "cohort": {"rows": len(frame), "primary_rows": int(primary.sum()), "warmup_rows": int(warmup.sum()),
                   "primary_months": months.tolist(),
                   "primary_source_tournament_year_proxies": int(
                       (primary_frame.tournament_name.fillna("unknown").astype(str) + "|" +
                        pd.to_datetime(primary_frame.date).dt.year.astype(str)).nunique())},
        "metrics": metrics, "profiles": profiles, "favorite_calibration": favorite,
        "contrasts": contrasts, "guards": guards,
        "source_tournament_year_sensitivity": sensitivity, "warmup_disclosure": warmup_disclosure,
        "decision": "DEVELOPMENT_CANDIDATE_RETAINED" if primary_pass and guard_pass else "NOT_ADOPTED",
        "gates": {"primary_all_three": primary_pass, "all_eight_guards": guard_pass},
        "limitations": {"prospective_evidence": False, "full_cohort_is_not_primary": True,
                        "conditional_zero_support_unresolved": unresolved,
                        "market_diagnostic": "registered_optional_parent_owned"},
    })
